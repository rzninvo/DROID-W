import numpy as np
import torch
import lietorch

import src.geom.projective_ops as pops
from src.modules.droid_net import CorrBlock
from src.utils.mono_priors.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth
from src.utils.datasets import load_metric_depth, load_img_feature
from src.utils.mono_priors.img_feature_extractors import predict_img_features, get_feature_extractor

class MotionFilter:
    """ This class is used to filter incoming frames and extract features
        mainly inherited from DROID-SLAM
    """

    def __init__(self, net, video, cfg, thresh=2.5, device="cuda:0"):
        self.cfg = cfg
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

        self.uncertainty_aware = cfg['tracking']["uncertainty_params"]['activate']
        self.save_dir = cfg['data']['output'] + '/' + cfg['scene']
        self.metric_depth_estimator = get_metric_depth_estimator(cfg)
        if cfg['mapping']["uncertainty_params"]['activate']:
            # If mapping needs dino features, we still need feature extractor
            self.feat_extractor = get_feature_extractor(cfg)

        # Separate CUDA stream for mono prior inference (depth + DINO)
        self.depth_stream = torch.cuda.Stream(device=device)

        # Segmentation for dynamic object masking
        seg_cfg = cfg.get('mono_prior', {}).get('segmentation', {})
        self.use_segmentation = seg_cfg.get('activate', False)
        if self.use_segmentation:
            from src.utils.mono_priors.seg_model import get_detector, DYNAMIC_CLASSES
            dynamic_classes = seg_cfg.get('dynamic_classes', list(DYNAMIC_CLASSES))
            self.seg_detector = get_detector(
                model_name=cfg['mono_prior'].get('detector', 'yolov8s-worldv2.pt'),
                classes=dynamic_classes,
                device=device,
            )
            self.seg_dynamic_classes = set(dynamic_classes)
            self.seg_conf_thresh = seg_cfg.get('conf_thresh', 0.15)

    @torch.amp.autocast('cuda',enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @torch.amp.autocast('cuda',enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, intrinsics=None):
        """ main update operation - run on every frame in video """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // self.video.down_scale
        wd = image.shape[-1] // self.video.down_scale

        # normalize images
        inputs = image[None, :, :].to(self.device)
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)       # [1, 128, 45, 80]

        force_to_add_keyframe = False

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            # Run depth estimation on separate stream, overlapping with context encoding
            with torch.cuda.stream(self.depth_stream):
                mono_depth = predict_metric_depth(self.metric_depth_estimator,tstamp,image,self.cfg,self.device,save_depth=(self.cfg['mono_prior']['save_depth'] or self.cfg['mapping']["enable"]))
            # Context encoding on default stream (runs concurrently with depth)
            net, inp = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            # Sync depth stream before using mono_depth
            self.depth_stream.synchronize()
            if self.uncertainty_aware:
                dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=self.cfg['mono_prior']['save_feature'])
            else:
                dino_features = None
                if self.cfg['mapping']["uncertainty_params"]['activate']:
                    _ = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=True)
            self.video.append(tstamp, image[0], Id, 1.0, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0,0], inp[0,0], dino_features)
            if self.use_segmentation:
                self._update_seg_mask(self.video.counter.value - 1, image)
        ### only add new frame if there is enough motion ###
        else:
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            if self.cfg['tracking']['force_keyframe_every_n_frames'] > 0:
                # Actually, tstamp is the frame idx
                last_tstamp = self.video.timestamp[self.video.counter.value-1]
                force_to_add_keyframe = (tstamp - last_tstamp) >= self.cfg['tracking']['force_keyframe_every_n_frames']


            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh or force_to_add_keyframe:
                self.count = 0
                # Run depth estimation on separate stream, overlapping with context encoding
                with torch.cuda.stream(self.depth_stream):
                    mono_depth = predict_metric_depth(self.metric_depth_estimator,tstamp,image,self.cfg,self.device,save_depth=(self.cfg['mono_prior']['save_depth'] or self.cfg['mapping']["enable"]))
                # Context encoding on default stream (runs concurrently with depth)
                net, inp = self.__context_encoder(inputs[:,[0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                # Sync depth stream before using mono_depth
                self.depth_stream.synchronize()
                if self.uncertainty_aware:
                    dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=self.cfg['mono_prior']['save_feature'])
                else:
                    dino_features = None
                    if self.cfg['mapping']["uncertainty_params"]['activate']:
                        _ = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,save_feat=True)
                # add new frame to video, all params
                self.video.append(tstamp, image[0], None, None, mono_depth, intrinsics / float(self.video.down_scale), gmap, net[0], inp[0], dino_features)
                # gmap: torch.Size([1, 128, 45, 80]) net[0]: [128, 45, 80] inp: [1, 128, 45, 80], dino_features: [25, 45, 384]
                if self.use_segmentation:
                    self._update_seg_mask(self.video.counter.value - 1, image)
            else:
                self.count += 1

        return force_to_add_keyframe

    @torch.no_grad()
    def _update_seg_mask(self, idx, image_tensor):
        """Run YOLO-World and write a static/dynamic mask into video.seg_masks."""
        from src.utils.mono_priors.seg_model import detect_objects, create_dynamic_mask

        # image_tensor is [C, H, W] float in [0, 1] RGB  (or [1, C, H, W])
        img = image_tensor
        if img.dim() == 4:
            img = img[0]
        # Convert to uint8 HWC numpy for YOLO
        image_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        detections = detect_objects(self.seg_detector, image_np, conf_thresh=self.seg_conf_thresh)

        ht = self.video.ht // self.video.down_scale
        wd = self.video.wd // self.video.down_scale
        mask_np = create_dynamic_mask(detections, ht, wd, self.seg_dynamic_classes)
        self.video.seg_masks[idx] = torch.from_numpy(mask_np).to(self.device)

    @torch.no_grad()
    def get_img_feature(self, tstamp, image, suffix=''):
        dino_features = predict_img_features(self.feat_extractor,tstamp,image,self.cfg,self.device,suffix=suffix,save_feat=self.cfg['mono_prior']['save_feature'])
        return dino_features
