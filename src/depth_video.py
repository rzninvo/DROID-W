import numpy as np
import torch
import lietorch
import droid_backends
import src.geom.ba
from torch.multiprocessing import Value
from torch.multiprocessing import Lock
import torch.nn.functional as F

from src.modules.droid_net import cvx_upsample
import src.geom.projective_ops as pops
from src.utils.common import align_scale_and_shift
from src.utils.Printer import FontColor
from src.utils.dyn_uncertainty import mapping_utils as map_utils
from src.utils.plot_utils import create_gif_from_directory
import matplotlib.pyplot as plt
import os
from src.utils.sys_timer import timer
import PIL
import PIL.Image as Image
from sklearn.decomposition import PCA
from tqdm import tqdm

class DepthVideo:
    ''' store the estimated poses and depth maps, 
        shared between tracker and mapper '''
    def __init__(self, cfg, printer):
        self.cfg =cfg
        self.output = f"{cfg['data']['output']}/{cfg['scene']}"
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd
        self.counter = Value('i', 0) # current keyframe count
        buffer = cfg['tracking']['buffer']
        self.printer = printer
        self.metric_depth_reg = cfg['tracking']['backend']['metric_depth_reg']
        if not self.metric_depth_reg:
            self.printer.print(f"Metric depth for regularization is not activated.",FontColor.INFO)
            self.printer.print(f"This should not happen for WildGS-SLAM unless you are doing ablation study",FontColor.INFO)
        self.mono_thres = cfg['tracking']['mono_thres']
        self.device = cfg['device']
        self.down_scale = 8
        self.slice_h = slice(self.down_scale // 2 - 1, ht//self.down_scale*self.down_scale+1, self.down_scale)
        self.slice_w = slice(self.down_scale // 2 - 1, wd//self.down_scale*self.down_scale+1, self.down_scale)
        ### state attributes ###
        self.timestamp = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
        # To save gpu ram, we put images to cpu as it is never used
        self.images = torch.zeros(buffer, 3, ht, wd, device='cpu', dtype=torch.float32)

        # whether the valid_depth_mask is calculated/updated, if dirty, not updated, otherwise, updated
        self.dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_() 
        # whether the corresponding part of pointcloud is deformed w.r.t. the poses and depths 
        self.npc_dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()

        self.poses = torch.zeros(buffer, 7, device=self.device, dtype=torch.float).share_memory_()  # world to camera
        self.disps = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.zeros = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.depth_scale = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.depth_shift = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.valid_depth_mask = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()
        self.valid_depth_mask_small = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.bool).share_memory_()
        ### vlm-mahbod: ViPE-style dynamic-object BA masking (ViPE paper 3.2.4:
        ### static mask M multiplies the flow-network weight). Opt-in via env
        ### DROIDW_DYN_MASKS=<npz>: keys = str(stream frame idx), values =
        ### (H,W) uint8 dynamic-union masks (1 = movable object). Frames absent
        ### from the npz fall back to the nearest masked frame within +-8, else
        ### unmasked. Default (env unset) = zero masks = canonical behavior.
        self.dyn_mask_lookup = None
        self.dyn_masks8 = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        _dm_path = os.environ.get('DROIDW_DYN_MASKS', '')
        if _dm_path:
            _dm_npz = np.load(_dm_path)
            self.dyn_mask_lookup = {int(k): np.asarray(_dm_npz[k]) for k in _dm_npz.files}
            print(f"[dyn-mask] loaded {len(self.dyn_mask_lookup)} frame masks from {_dm_path}")
        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, 1, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.debug = cfg['debug']

        self.uncertainty_aware = cfg['tracking']["uncertainty_params"]['activate']
        self.enable_bidirectional_uncer = cfg['tracking']["uncertainty_params"]['enable_bidirectional_uncer']
        if self.uncertainty_aware:
            n_features = self.cfg["tracking"]["uncertainty_params"]['feature_dim']

            # This check is to ensure the size of self.dino_feats
            if self.cfg["mono_prior"]["feature_extractor"] not in ["dinov2_reg_small_fine", "dinov2_small_fine","dinov2_vits14", "dinov2_vits14_reg", "dinov3_vits16", "dinov3_vits16plus"]:
                raise ValueError("You are using a new feature extractor, make sure the downsample factor is 14")
            if self.cfg["mono_prior"]["feature_extractor"] in ["dinov3_vits16", "dinov3_vits16plus"]:
                self.feature_downsample_factor = 16
            else:
                self.feature_downsample_factor = 14
            
            # The followings are in cpu to save memory
            self.dino_feats = torch.zeros(buffer, ht//self.feature_downsample_factor, wd//self.feature_downsample_factor, n_features, device='cpu', dtype=torch.float).share_memory_()
            self.dino_feats_resize = torch.zeros(buffer, n_features, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
            self.uncertainties = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
            # use kaiming normal initialization to initialize the affine weights
            # [1, n_features + 1] ：first n_features are weights, last one is bias
            self.affine_weights = torch.empty((1, n_features + 1), dtype=torch.float, device=self.device)
            # use kaiming normal initialization to initialize the affine weights
            torch.nn.init.kaiming_normal_(self.affine_weights[:, :-1], mode='fan_in', nonlinearity='linear')
            # set bias to 0
            self.affine_weights[:, -1].zero_()
            self.affine_weights = self.affine_weights.squeeze(0).share_memory_()
            self.enable_affine_transform = cfg['tracking']['uncertainty_params']['enable_affine_transform']
            self.temp_y_cdot = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        else:
            self.dino_feats = None
            self.dino_feats_resize = None

        # HERMES variant: per-keyframe temporal stability S(u) buffer
        # (RADIO-ViPE ARK port, see src/utils/dyn_uncertainty/temporal_stability.py).
        self.stability_enabled = cfg['tracking']['stability']['enable']
        if self.stability_enabled:
            self.stability = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        else:
            self.stability = None

    def get_lock(self):
        return self.counter.get_lock()

    def _set_dyn_mask(self, index, tstamp):
        """vlm-mahbod: store the (H,W) dynamic mask of stream frame `tstamp`
        into the 1/8-grid buffer slot `index` (area-downsample, >0.3 -> 1).
        Nearest masked frame within +-8 covers pass-2 keyframe drift; no match
        -> unmasked (zeros) so behavior degrades to canonical, never breaks."""
        m = self.dyn_mask_lookup.get(tstamp)
        if m is None:
            for d in range(1, 9):
                if tstamp - d in self.dyn_mask_lookup:
                    m = self.dyn_mask_lookup[tstamp - d]; break
                if tstamp + d in self.dyn_mask_lookup:
                    m = self.dyn_mask_lookup[tstamp + d]; break
        if m is None:
            self.dyn_masks8[index] = 0.0
            return
        mt = torch.from_numpy(m.astype(np.float32))[None, None].to(self.device)
        # replicate the dataset transform (datasets.py get_color): resize the
        # raw frame to (H_out+2*H_edge, W_out+2*W_edge) then crop the edges —
        # otherwise the mask is scaled/shifted on edge-cropped configs (TUM
        # uses H_edge=W_edge=8; Bonn uses 0, where this reduces to a resize).
        _cam = self.cfg.get('cam', {}) if hasattr(self, 'cfg') else {}
        _he = int(_cam.get('H_edge', 0) or 0)
        _we = int(_cam.get('W_edge', 0) or 0)
        mt = F.interpolate(mt, size=(self.ht + 2 * _he, self.wd + 2 * _we), mode='area')
        if _he > 0:
            mt = mt[:, :, _he:-_he, :]
        if _we > 0:
            mt = mt[:, :, :, _we:-_we]
        m8 = F.interpolate(mt, size=(self.ht // self.down_scale, self.wd // self.down_scale), mode='area')[0, 0]
        self.dyn_masks8[index] = (m8 > 0.3).float()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.timestamp[index] = item[0]
        self.images[index] = item[1].cpu()

        # vlm-mahbod: attach the frame's dynamic mask (by stream index) at
        # insertion so BA sees it wherever this keyframe lands in the buffer.
        # index may be int, Tensor, slice, or list (trajectory_filler uses
        # slices); normalize both index and timestamps to flat lists.
        if self.dyn_mask_lookup is not None:
            _ts_arr = torch.as_tensor(item[0]).reshape(-1)
            if isinstance(index, torch.Tensor):
                _idx_arr = index.reshape(-1).tolist()
            elif isinstance(index, slice):
                _idx_arr = list(range(*index.indices(self.dyn_masks8.shape[0])))
            elif isinstance(index, (list, tuple)):
                _idx_arr = [int(v) for v in index]
            else:
                _idx_arr = [int(index)]
            if _ts_arr.numel() == 1 and len(_idx_arr) > 1:
                _ts_arr = _ts_arr.repeat(len(_idx_arr))
            if _ts_arr.numel() == len(_idx_arr):
                for _bi, _tv in zip(_idx_arr, _ts_arr.tolist()):
                    self._set_dyn_mask(int(_bi), int(_tv))
            else:
                print(f"[WARN] dyn-mask: index/timestamp count mismatch "
                      f"({len(_idx_arr)} vs {_ts_arr.numel()}), fallback=skip attach")

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]


        if item[4] is not None:
            mono_depth = item[4][self.slice_h,self.slice_w]
            self.mono_disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)
            self.mono_disps_up[index] = torch.where(item[4]>0, 1.0/item[4], 0)
            # self.disps[index] = torch.where(mono_depth>0, 1.0/mono_depth, 0)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6 and item[6] is not None:
            self.fmaps[index] = item[6]

        if len(item) > 7 and item[7] is not None:
            self.nets[index] = item[7]

        if len(item) > 8 and item[8] is not None:
            self.inps[index] = item[8]

        if len(item) > 9 and item[9] is not None:
            self.dino_feats[index] = item[9].cpu()

            if len(item[9].shape) == 3:
                self.dino_feats_resize[index] = F.interpolate(item[9].permute(2,0,1).unsqueeze(0),
                                                            self.disps_up.shape[-2:], 
                                                            mode='bilinear').squeeze()[:,self.slice_h,self.slice_w]
                # y_cdot = w * x + b
                # y = log(1 + exp(y_cdot))
                if self.enable_affine_transform:
                    y_cdot = self.dino_feats_resize[index].permute(1,2,0) @ self.affine_weights[:-1] + self.affine_weights[-1]
                    self.temp_y_cdot[index] = y_cdot
                    self.uncertainties[index] = torch.log(1.1 + torch.exp(y_cdot))
            else:
                self.dino_feats_resize[index] = F.interpolate(item[9].permute(0,3,1,2),
                                                            self.disps_up.shape[-2:], 
                                                            mode='bilinear')[:,:,self.slice_h,self.slice_w]
                # y_cdot = w * x + b
                # y = log(1 + exp(y_cdot))
                if self.enable_affine_transform:
                    y_cdot = self.dino_feats_resize[index].permute(0, 2, 3, 1) @ self.affine_weights[:-1] + self.affine_weights[-1]
                    self.temp_y_cdot[index] = y_cdot
                    self.uncertainties[index] = torch.log(1.1 + torch.exp(y_cdot))
                    

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    def init_w_mono_disp(self, start_idx, end_idx):
        with self.get_lock():
            self.disps[start_idx:end_idx] = self.mono_disps[start_idx:end_idx]
            self.disps_up[start_idx:end_idx] = self.mono_disps_up[start_idx:end_idx]

    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def upsample_weight(self, weight):
        """ upsample weight to the original image size """
        weight_up = F.interpolate(weight.unsqueeze(0), 
                                size=(self.ht, self.wd), mode='bilinear').squeeze()
        return weight_up

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.set_dirty(0,self.counter.value)


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N),indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d
    
    def project_images_with_mask(self, images, pixel_positions, masks=None):
        """ 
            Project images/depths from the input pixel positions using bilinear interpolation.
            This function will automatically return the mask where the given pixel positions are out of the images
        Args:
            images (torch.Tensor): A tensor of shape [B, C, H, W] representing the images/depths.
            pixel_positions (torch.Tensor): A tensor of shape [B, H, W, 2] containing float 
                                            pixel positions for interpolation. Note that [:,:,:,0]
                                            is width and [:,:,:,1] is height.
            masks (torch.Tensor, optional): A boolean tensor of shape [B, H, W]. If provided, 
                                            specifies valid pixels. Default is None, which 
                                            results in all pixels being valid at the begining.
        
        Returns:
            torch.Tensor: A tensor of shape [B, C, H, W] containing the projected images/depths, 
                        where invalid pixels are set to 0.
            torch.Tensor: The combined mask that filters out invalid positions and applies
                      the original mask.
        """
        B, C, H, W = images.shape
        device = images.device

        # If masks are not provided, create a mask of all ones (True) with the same shape as the images
        if masks is None:
            masks = torch.ones(B, H, W, dtype=torch.bool, device=device)
        
        # Normalize pixel positions to range [-1, 1]
        grid = pixel_positions.clone()
        grid[..., 0] = 2.0 * (grid[..., 0] / (W - 1)) - 1.0
        grid[..., 1] = 2.0 * (grid[..., 1] / (H - 1)) - 1.0

        projected_image = F.grid_sample(images, grid, mode='bilinear', align_corners=True)

        # Mask out invalid positions where x or y are out of bounds and combine it with the initial mask
        valid_mask = (pixel_positions[..., 0] >= 0) & (pixel_positions[..., 0] < W - 1) & \
                    (pixel_positions[..., 1] >= 0) & (pixel_positions[..., 1] < H - 1)
        valid_mask &= masks

        # Apply the combined mask: set to 0 where combined mask is False
        projected_image = projected_image.permute(0, 2, 3, 1)  # conver to [B, H, W, C]
        projected_image = projected_image * valid_mask.unsqueeze(-1)
        
        return projected_image.permute(0, 3, 1, 2), valid_mask  # Return to [B, C, H, W]

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1, gamma=0.02, tao=0.1, lr=1e-2, weight_decay=2e-4,
           motion_only=False, enable_update_uncer=False, enable_udba=False, visualization_stage=False):      # ii, jj represent all img pairs
        
        with self.get_lock():
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            target = target.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()

            # if there is NaN of inf value for self.affine_weights, assert
            if self.uncertainty_aware:
                assert not torch.isnan(self.affine_weights).any(), "self.affine_weights has NaN value"
                assert not torch.isinf(self.affine_weights).any(), "self.affine_weights has inf value"
            if not self.metric_depth_reg:
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,           # the shape of poses is determined by buffer size
                    target, weight, self.uncertainties, 
                    self.temp_y_cdot,
                    self.dino_feats_resize,
                    self.affine_weights,
                    eta, ii, jj, t0, t1, iters, lm, ep, 
                    self.cfg['tracking']['uncertainty_params']['gamma_data'], 
                    self.cfg['tracking']['uncertainty_params']['gamma_prior'], 
                    self.cfg['tracking']['uncertainty_params']['gamma_depth'],
                    lr, weight_decay,
                    motion_only, False, enable_update_uncer,
                    enable_udba, self.enable_affine_transform,
                    self.enable_bidirectional_uncer,
                    self.debug)         # poses: [buffer, 7], disps: [buffer, h, w], 
            else:
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.mono_disps,
                    target, weight, self.uncertainties, 
                    self.temp_y_cdot,
                    self.dino_feats_resize,
                    self.affine_weights,
                    eta, ii, jj, t0, t1, iters, lm, ep, 
                    self.cfg['tracking']['uncertainty_params']['gamma_data'], 
                    self.cfg['tracking']['uncertainty_params']['gamma_prior'], 
                    self.cfg['tracking']['uncertainty_params']['gamma_depth'],
                    lr, weight_decay,
                    motion_only, False, enable_update_uncer,
                    enable_udba, self.enable_affine_transform,
                    self.enable_bidirectional_uncer,
                    self.debug)          # t0, t1: window of keyframes for BA
            
            self.disps.clamp_(min=1e-5)

    @torch.no_grad()
    def visualize_uncertainty(self, target, weight, ii, jj, frame_choice="nearest", mode="Before"):
        """ 
        visualize the uncertainty before and after optimization, reprojection error, and the weight prediction
        """
        # for ind, i, j in zip(range(ii.shape[0]), ii, jj):
        i = ii.max().item()
        mask = (ii == i)                       # bool mask, mark all the same max ii
        idx_nd = mask.nonzero(as_tuple=False)  # [K, ndim]

        # 2) get all candidates of j
        j_candidates = jj[mask]                # [K]

        # 3) select the max j from the candidates and get the relative index
        if frame_choice == "nearest":
            j, rel = j_candidates.max(dim=0)       # rel is the index in j_candidates
        elif frame_choice == "farthest":
            j, rel = j_candidates.min(dim=0)
        elif frame_choice == "random":
            rand_idx = torch.randint(0, j_candidates.shape[0], (1,))
            j = j_candidates[rand_idx].item()
            rel = rand_idx
        rel = rel.item()

        # 4) output the max j and the corresponding original index
        ind = idx_nd[rel].item()

        weight_pred = self.upsample_weight(weight[ind].squeeze()).cpu().numpy()

        img_i = self.images[i].permute(1,2,0).numpy()
        img_j = self.images[j].permute(1,2,0).numpy()

        # compute the reprojection error between img_i and img_j
        reprojected_coords, valid_mask = self.reproject(i, j)
        reprojection_error = target[ind].permute(1,2,0) - reprojected_coords.squeeze()
        reprojection_error_x = reprojection_error[:,:,0].abs()
        reprojection_error_y = reprojection_error[:,:,1].abs()
        reprojection_error_norm = torch.norm(reprojection_error, dim=-1)

        reprojection_error_x = self.upsample_weight(reprojection_error_x.unsqueeze(0)).cpu().numpy()
        reprojection_error_y = self.upsample_weight(reprojection_error_y.unsqueeze(0)).cpu().numpy()
        reprojection_error_norm = self.upsample_weight(reprojection_error_norm.unsqueeze(0)).cpu().numpy()

        disp_i = self.upsample_weight(self.disps[i].unsqueeze(0)).cpu().numpy()
        mono_disp_i = self.upsample_weight(self.mono_disps[i].unsqueeze(0)).cpu().numpy()

        # compute DINO features similarity between img_i and img_j
        img_w = self.wd // self.down_scale
        img_h = self.ht // self.down_scale
        dino_feats_i = F.normalize(self.dino_feats_resize[i], p=2, dim=1).unsqueeze(0)   # [1, C, H, W]
        dino_feats_j = F.normalize(self.dino_feats_resize[j], p=2, dim=1).unsqueeze(0)   # [1, C, H, W]
        dino_feats_reproj, valid_mask = self.project_images_with_mask(dino_feats_j, reprojected_coords.resize(1, img_h, img_w, 2))
        dino_feats_reproj = F.normalize(dino_feats_reproj, p=2, dim=1)  # Normalize features for cosine similarity
        dino_feats_similarity_reproj = (dino_feats_i * dino_feats_reproj).sum(dim=1)  # [1, H, W]
        dino_feats_similarity_reproj = self.upsample_weight(dino_feats_similarity_reproj).cpu().numpy()

        # Create the figure WITH constrained layout
        fig_height = 12
        fig_width = 13
        aspect_ratio = img_i.shape[1] / img_i.shape[0]
        fig_width = fig_width * aspect_ratio
        fig, axes = plt.subplots(3, 4, figsize=(fig_width, fig_height), constrained_layout=True)

        # vis image i
        axes[0,0].imshow(img_i)
        axes[0,0].set_title("Image i Visualization")
        # visualize img_j
        axes[0,1].imshow(img_j)
        axes[0,1].set_title("Image j Visualization")

        # visualize the weight (X/Y)
        wmax = weight_pred.max()
        axes[0,2].imshow(img_i)
        im1 = axes[0,2].imshow(weight_pred[0], cmap='jet', alpha=0.7, vmin=0, vmax=wmax)
        axes[0,2].set_title("X Weight Pred")
        fig.colorbar(im1, ax=axes[0,2], fraction=0.046, pad=0.04)

        axes[0,3].imshow(img_i)
        im2 = axes[0,3].imshow(weight_pred[1], cmap='jet', alpha=0.7, vmin=0, vmax=wmax)
        axes[0,3].set_title("Y Weight Pred")
        fig.colorbar(im2, ax=axes[0,3], fraction=0.046, pad=0.04)


        uncer_pred_i = self.uncertainties[i]
        uncer_rescaled = torch.clamp(45.0 * uncer_pred_i - 35.0, min=0.1)
        mask_i = torch.clamp(1.0 / uncer_rescaled, min=0.0, max=1.0)
        im9 = axes[1,0].imshow(mask_i.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        axes[1,0].set_title("Optimized Dynamic Mask of Frame i")

        # vis DINO feature similarity at (2, 2)
        im8 = axes[1,1].imshow(dino_feats_similarity_reproj, cmap='jet', vmin=0, vmax=1)
        axes[1,1].set_title("DINO Feature Similarity Between i and j")
        fig.colorbar(im8, ax=axes[1,1], fraction=0.046, pad=0.04)

        # save dino feats similarity
        os.makedirs(f"{self.output}/intermediate_results", exist_ok=True)
        dino_feats_similarity_reproj_vis = plt.get_cmap('viridis')(dino_feats_similarity_reproj)
        Image.fromarray((dino_feats_similarity_reproj_vis * 255.0).astype(np.uint8)).save(f"{self.output}/intermediate_results/dino_feats_similarity_reproj_{i:03d}_vs_{j:03d}.png")

        """
        # visualize high-dimensional features using PCA projection
        """

        def visualize_dino_feature_pca(features, save_path=None, scale_each=False):
            """
            Visualize DINO features (C, H, W) or (B, C, H, W) as RGB image using PCA projection.
            Args:
                features: torch.Tensor or np.ndarray, shape [C,H,W] or [B,C,H,W]
                save_path: optional, if provided, save the RGB visualization
                scale_each: if True, scale per image when B>1
            """
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()

            # Handle batch
            if features.ndim == 4:
                feats_list = []
                for f in features:
                    feats_list.append(_pca_to_rgb_single(f, scale_each))
                vis = np.concatenate(feats_list, axis=1)  # horizontal concat
            elif features.ndim == 3:
                vis = _pca_to_rgb_single(features, scale_each)
            else:
                raise ValueError("Feature tensor must be [C,H,W] or [B,C,H,W]")

            plt.figure(figsize=(6,6))
            plt.imshow(vis)
            plt.axis('off')
            plt.title("DINO feature PCA→RGB projection")
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            # plt.show()
            return vis


        def _pca_to_rgb_single(feature, scale_each=False):
            """Helper: apply PCA→RGB"""
            C, H, W = feature.shape
            feat = feature.reshape(C, -1).T  # [HW, C]
            feat -= feat.mean(0, keepdims=True)

            pca = PCA(n_components=3)
            feat_rgb = pca.fit_transform(feat)  # [HW, 3]
            feat_rgb = feat_rgb.reshape(H, W, 3)

            # Normalize to [0,1]
            if scale_each:
                for i in range(3):
                    feat_rgb[..., i] = (feat_rgb[..., i] - feat_rgb[..., i].min()) / (feat_rgb[..., i].max() - feat_rgb[..., i].min() + 1e-6)
            else:
                feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min() + 1e-6)

            return feat_rgb

        dino_feats_i_vis = visualize_dino_feature_pca(dino_feats_i)
        dino_feats_j_vis = visualize_dino_feature_pca(dino_feats_j)
        # from [H, W, C] to [C, H, W] for numpy array.
        dino_feats_i_vis = dino_feats_i_vis.transpose(2,0,1)
        dino_feats_j_vis = dino_feats_j_vis.transpose(2,0,1)
        dino_feats_i_vis_upsampled = F.interpolate(torch.from_numpy(dino_feats_i_vis).unsqueeze(0), size=(self.ht, self.wd), mode='bilinear', align_corners=False)
        dino_feats_j_vis_upsampled = F.interpolate(torch.from_numpy(dino_feats_j_vis).unsqueeze(0), size=(self.ht, self.wd), mode='bilinear', align_corners=False)
        # from [C, H, W] to [H, W, C]
        dino_feats_i_vis_upsampled = dino_feats_i_vis_upsampled.squeeze(0).cpu().numpy().transpose(1,2,0)
        dino_feats_j_vis_upsampled = dino_feats_j_vis_upsampled.squeeze(0).cpu().numpy().transpose(1,2,0)
        Image.fromarray((dino_feats_i_vis_upsampled * 255.0).astype(np.uint8)).save(f"{self.output}/intermediate_results/dino_feats_i_{i:03d}.png")
        Image.fromarray((dino_feats_j_vis_upsampled * 255.0).astype(np.uint8)).save(f"{self.output}/intermediate_results/dino_feats_j_{j:03d}.png")

        # save high-resolution mono disparity
        mono_disp_i_high_res = self.mono_disps_up[i]
        mono_disp_j_high_res = self.mono_disps_up[j]
        mono_disp_i_high_res = torch.clamp(mono_disp_i_high_res, min=0.1, max=1)
        i_min = mono_disp_i_high_res.min()
        i_max = mono_disp_i_high_res.max()
        mono_disp_i_high_res = (mono_disp_i_high_res - i_min) / (i_max - i_min)
        mono_disp_j_high_res = torch.clamp(mono_disp_j_high_res, min=0.1, max=1)
        j_min = mono_disp_j_high_res.min()
        j_max = mono_disp_j_high_res.max()
        mono_disp_j_high_res = (mono_disp_j_high_res - j_min) / (j_max - j_min)
        mono_disp_i_high_res_vis = plt.get_cmap('viridis')(mono_disp_i_high_res.cpu().numpy())
        mono_disp_j_high_res_vis = plt.get_cmap('viridis')(mono_disp_j_high_res.cpu().numpy())
        Image.fromarray((mono_disp_i_high_res_vis * 255.0).astype(np.uint8)).save(f"{self.output}/intermediate_results/mono_disp_i_{i:03d}.png")
        Image.fromarray((mono_disp_j_high_res_vis * 255.0).astype(np.uint8)).save(f"{self.output}/intermediate_results/mono_disp_j_{j:03d}.png")

        # vis mono disparity map of image i
        im6 = axes[1,2].imshow(mono_disp_i, cmap='viridis')
        axes[1,2].set_title("Mono Disparity of Image i")
        fig.colorbar(im6, ax=axes[1,2], fraction=0.046, pad=0.04)

        # vis depth map of image i at (2, 1)
        im7 = axes[1,3].imshow(disp_i, cmap='viridis')
        axes[1,3].set_title("Optimized Disparity of Image i")
        fig.colorbar(im7, ax=axes[1,3], fraction=0.046, pad=0.04)

        # vis optimized uncertainty of image i with contours
        im_uncer = axes[2,0].imshow(uncer_pred_i.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
        axes[2,0].set_title("Optimized Uncertainty of Image i")
        fig.colorbar(im_uncer, ax=axes[2,0], fraction=0.046, pad=0.04)
        contours = axes[2,0].contour(uncer_pred_i.cpu().numpy(), levels=10, colors='black', linewidths=0.5)
        axes[2,0].clabel(contours, inline=True, fontsize=8)
        axes[2,1].imshow(uncer_rescaled.cpu().numpy(), cmap='jet', vmin=0, vmax=10.0)
        axes[2,1].set_title("Rescaled Uncertainty of Frame i")

        # visualize photometric error
        # downsample the image to the same size as the reprojected coordinates
        img_i_downsampled = F.interpolate(self.images[i].unsqueeze(0), size=(img_h, img_w), mode='bilinear', align_corners=False)
        img_j_downsampled = F.interpolate(self.images[j].unsqueeze(0), size=(img_h, img_w), mode='bilinear', align_corners=False)
        img_j_warp, valid_mask = self.project_images_with_mask(img_j_downsampled.to(self.device), reprojected_coords.resize(1, img_h, img_w, 2))
        photometric_error = (img_i_downsampled.to(self.device) - img_j_warp) * valid_mask
        photometric_error = photometric_error.abs().squeeze(0)
        photometric_error = photometric_error.permute(1,2,0).cpu().numpy()        # [H, W, 3]
        axes[2,2].imshow(photometric_error)
        axes[2,2].set_title("Photometric Error")

        axes[2,3].imshow(img_i)
        im5 = axes[2,3].imshow(reprojection_error_norm, cmap='jet', alpha=0.7)
        mean_reprojection_error_norm = reprojection_error_norm.mean()
        axes[2,3].set_title(f"Reprojection Error Norm (Mean: {mean_reprojection_error_norm:.4f})")
        fig.colorbar(im5, ax=axes[2,3], fraction=0.046, pad=0.04)

        for ax in axes.ravel():
            ax.axis("off")

        fig.suptitle(f"Keyframe {i:03d} vs {j:03d}", fontsize=16)

        os.makedirs(f"{self.output}/intermediate_results", exist_ok=True)
        fig.savefig(f"{self.output}/intermediate_results/ts_{int(self.timestamp[i]):05d}_kf_{i:03d}_vs_kf_{j:03d}_{mode}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    @torch.no_grad()
    def visualize_all_opt_params(self, out_directory=None, iteration="final"):
        """ 
        visualize the uncertainty before and after optimization, disparity map
        """

        plot_dir = os.path.join(out_directory, "plots_" + iteration)
        # add tqdm progress bar
        for idx in tqdm(range(self.counter.value), desc="Visualizing all optimized parameters", total=self.counter.value):
            img_i = self.images[idx].permute(1,2,0).cpu().numpy()

            uncer_pred = self.uncertainties[idx]
            uncer_rescaled = torch.clamp(45.0 * uncer_pred - 35.0, min=0.1)

            mask_i = torch.clamp(1.0 / uncer_rescaled, min=0.0, max=1.0)

            # Create the figure WITH constrained layout
            plot_cols = 4
            plot_rows = 2
            fig_height = plot_rows * 4 + 0.5    # leave space for the title
            fig_width = plot_cols * 4
            aspect_ratio = img_i.shape[1] / img_i.shape[0]
            fig_width = fig_width * aspect_ratio
            fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(fig_width, fig_height), constrained_layout=True)

            # vis image i
            axes[0,0].imshow(img_i)
            axes[0,0].set_title("Input Image i")

            # visualize uncertainty heatmap with contours at (row 1, col 0)
            im_unc = axes[1,0].imshow(uncer_pred.cpu().numpy(), cmap='jet', vmin=0.0, vmax=1.0)
            contours = axes[1,0].contour(uncer_pred.cpu().numpy(), levels=10, colors='black', linewidths=0.5)
            axes[1,0].clabel(contours, inline=True, fontsize=8)
            axes[1,0].set_title("Uncertainty")

            # rescaled uncertainty
            axes[0,1].imshow(uncer_rescaled.cpu().numpy(), cmap='jet', vmin=0, vmax=10.0)
            axes[0,1].set_title("Rescaled Uncertainty")

            # visualize high-resolution scaled uncertainty
            # uncer_rescaled_high_res = self.upsample_weight(uncer_rescaled.unsqueeze(0))
            uncer_pred_high_res = self.upsample_weight(uncer_pred.unsqueeze(0))
            uncer_rescaled_high_res = torch.clamp(45.0 * uncer_pred_high_res - 35.0, min=0.1)
            axes[1,1].imshow(uncer_rescaled_high_res.cpu().numpy(), cmap='jet', vmin=0, vmax=10.0)
            axes[1,1].set_title("High-Resolution Scaled Uncertainty")

            # vis mono disparity map of image i
            mono_disp = self.mono_disps[idx].cpu().numpy()
            vmax = min(mono_disp.max().item(), 5.0)
            im2 = axes[0,2].imshow(mono_disp, cmap='viridis', vmin=0, vmax=vmax)
            axes[0,2].set_title("Mono Disparity")

            # visualize high-resolution mono disparity
            mono_disp_high_res = self.mono_disps_up[idx]
            vmax = min(mono_disp_high_res.max().item(), 5.0)
            axes[1,2].imshow(mono_disp_high_res.cpu().numpy(), cmap='viridis', vmin=0, vmax=vmax)
            axes[1,2].set_title("High-Resolution Mono Disparity")

            # vis depth map of image i at (2, 1)
            droid_disp = self.disps[idx].cpu().numpy()
            vmax = min(droid_disp.max().item(), 5.0)
            im3 = axes[0,3].imshow(droid_disp, cmap='viridis', vmin=0, vmax=vmax)
            axes[0,3].set_title("Optimized Disparity")

            # visualize high-resolution optimized disparity
            droid_disp_high_res = self.disps_up[idx]
            vmax = min(droid_disp_high_res.max().item(), 5.0)
            axes[1,3].imshow(droid_disp_high_res.cpu().numpy(), cmap='viridis', vmin=0, vmax=vmax)
            axes[1,3].set_title("High-Resolution Optimized Disparity")

            for ax in axes.ravel():
                ax.axis("off")

            fig.suptitle(f"Keyframe idx {idx:03d}, Frame idx {int(self.timestamp[idx]):05d}", fontsize=20)

            os.makedirs(f"{plot_dir}", exist_ok=True)
            fig.savefig(f"{plot_dir}/video_kf_{idx:03d}_ts_{int(self.timestamp[idx]):05d}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            # save the input image, scaled uncertainty, uncertainty with contours, and mask individually
            # Create separate directories for each image type
            input_dir = os.path.join(plot_dir, "input_images")
            uncer_dir = os.path.join(plot_dir, "scaled_uncertainty")
            uncer_contour_dir = os.path.join(plot_dir, "uncertainty_contours")
            high_res_uncer_dir = os.path.join(plot_dir, "high_res_uncertainty")
            
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(uncer_dir, exist_ok=True)
            os.makedirs(uncer_contour_dir, exist_ok=True)
            os.makedirs(high_res_uncer_dir, exist_ok=True)

            def color_map(tensor, cmap='jet', vmin=0, vmax=1):
                return (plt.get_cmap(cmap)(tensor.cpu().numpy() / vmax)[:, :, :3] * 255.0).astype(np.uint8)
            
            # Save input image
            Image.fromarray((img_i * 255.0).astype(np.uint8)).save(f"{input_dir}/input_kf_{idx:03d}_ts_{int(self.timestamp[idx]):05d}.png")

            # Save scaled uncertainty
            uncer_rescaled_colored = color_map(uncer_rescaled, vmax=10.0)
            Image.fromarray(uncer_rescaled_colored).save(f"{uncer_dir}/uncertainty_kf_{idx:03d}_ts_{int(self.timestamp[idx]):05d}.png")
            
            # Save uncertainty with contours
            fig_contour, ax_contour = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            ax_contour.imshow(uncer_pred.cpu().numpy(), cmap='jet', vmin=0.0, vmax=1.0)
            contours = ax_contour.contour(uncer_pred.cpu().numpy(), levels=10, colors='black', linewidths=0.5)
            ax_contour.clabel(contours, inline=True, fontsize=8)
            ax_contour.axis("off")
            ax_contour.set_position([0, 0, 1, 1])  # Remove margins
            fig_contour.savefig(f"{uncer_contour_dir}/uncertainty_contour_kf_{idx:03d}_ts_{int(self.timestamp[idx]):05d}.png", dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig_contour)
            
            # Save high-resolution scaled uncertainty
            uncer_rescaled_high_res_colored = color_map(uncer_rescaled_high_res, vmax=10.0)
            Image.fromarray(uncer_rescaled_high_res_colored).save(f"{high_res_uncer_dir}/high_res_uncertainty_kf_{idx:03d}_ts_{int(self.timestamp[idx]):05d}.png")
            
        # Create gif
        create_gif_from_directory(plot_dir, plot_dir + '/output.gif', online=True)

    def get_depth_scale_and_shift(self,index, mono_depth:torch.Tensor, est_depth:torch.Tensor, weights:torch.Tensor):
        '''
        index: int
        mono_depth: [B,H,W]
        est_depth: [B,H,W]
        weights: [B,H,W]
        '''
        scale,shift,_ = align_scale_and_shift(mono_depth,est_depth,weights)
        self.depth_scale[index] = scale
        self.depth_shift[index] = shift
        return [self.depth_scale[index], self.depth_shift[index]]

    def get_pose(self,index,device):
        w2c = lietorch.SE3(self.poses[index].clone()).to(device) # Tw(droid)_to_c
        c2w = w2c.inv().matrix()  # [4, 4]
        return c2w

    def get_depth_and_pose(self,index,device):
        with self.get_lock():
            if self.metric_depth_reg:
                est_disp = self.mono_disps_up[index].clone().to(device)  # [h, w]
                est_depth = torch.where(est_disp>0.0, 1.0 / (est_disp), 0.0)
                depth_mask = torch.ones_like(est_disp,dtype=torch.bool).to(device)
                c2w = self.get_pose(index,device)
            else:
                est_disp = self.disps_up[index].clone().to(device)  # [h, w]
                est_depth = 1.0 / (est_disp)
                depth_mask = self.valid_depth_mask[index].clone().to(device)
                c2w = self.get_pose(index,device)
        return est_depth, depth_mask, c2w
    
    @torch.no_grad()
    def update_valid_depth_mask(self,up=True):
        '''
        For each pixel, check whether the estimated depth value is valid or not 
        by the two-view consistency check, see eq.4 ~ eq.7 in the paper for details

        up (bool): if True, check on the orignial-scale depth map
                   if False, check on the downsampled depth map
        '''
        if up:
            with self.get_lock():
                dirty_index, = torch.where(self.dirty.clone())
            if len(dirty_index) == 0:
                return
        else:
            curr_idx = self.counter.value-1
            dirty_index = torch.arange(curr_idx+1).to(self.device)
        # convert poses to 4x4 matrix
        disps = torch.index_select(self.disps_up if up else self.disps, 0, dirty_index)
        common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
        intrinsic = self.intrinsics[common_intrinsic_id].detach() * (self.down_scale if up else 1.0)
        depths = 1.0/disps
        thresh = self.cfg['tracking']['multiview_filter']['thresh'] * depths.mean(dim=[1,2]) 
        count = droid_backends.depth_filter(
            self.poses, self.disps_up if up else self.disps, intrinsic, dirty_index, thresh)
        filter_visible_num = self.cfg['tracking']['multiview_filter']['visible_num']
        multiview_masks = (count >= filter_visible_num) 
        depths[~multiview_masks]=torch.nan
        depths_reshape = depths.view(depths.shape[0],-1)
        depths_median = depths_reshape.nanmedian(dim=1).values
        masks = depths < 3*depths_median[:,None,None]
        if up:
            self.valid_depth_mask[dirty_index] = masks 
            self.dirty[dirty_index] = False
        else:
            self.valid_depth_mask_small[dirty_index] = masks 

    def set_dirty(self,index_start, index_end):
        self.dirty[index_start:index_end] = True
        self.npc_dirty[index_start:index_end] = True

    def save_video(self,path:str):
        poses = []
        for i in range(self.counter.value):
            depth, depth_mask, pose = self.get_depth_and_pose(i,'cpu')
            poses.append(pose)
        poses = torch.stack(poses,dim=0).numpy()

        timestamps = self.timestamp[:self.counter.value].cpu().numpy()
        images = self.images[:self.counter.value].cpu().numpy()
        tum_poses = self.poses[:self.counter.value].cpu().numpy()
        mono_disps = self.mono_disps_up[:self.counter.value].cpu().numpy()
        droid_disps_up = self.disps_up[:self.counter.value].cpu().numpy()
        droid_disps = self.disps[:self.counter.value].cpu().numpy()
        intrinsics = self.intrinsics[:self.counter.value].cpu().numpy()
        uncertainties = self.uncertainties[:self.counter.value].cpu().numpy()
        # HERMES variant: export per-keyframe temporal stability S(u) when enabled
        extra = {}
        if self.stability_enabled:
            extra['stability'] = self.stability[:self.counter.value].cpu().numpy()
        np.savez(path,
            timestamps=timestamps,
            images=images,
            poses=poses,tum_poses=tum_poses,
            mono_disps=mono_disps,
            droid_disps_up=droid_disps_up,
            droid_disps=droid_disps,
            intrinsics=intrinsics,
            uncertainties=uncertainties,
            **extra)
        self.printer.print(f"Saved final depth video: {path}",FontColor.INFO)

    def save_poses(self,path:str):
        poses = []
        timestamps = []
        for i in range(self.counter.value):
            _, _, pose = self.get_depth_and_pose(i,'cpu')
            timestamp = self.timestamp[i].cpu()
            poses.append(pose)
            timestamps.append(timestamp)
        poses = torch.stack(poses,dim=0).numpy()
        timestamps = torch.stack(timestamps,dim=0).numpy()   
        np.savez(path,poses=poses, timestamps=timestamps)
        self.printer.print(f"Saved poses and timestamp: {path}", FontColor.INFO)

    def eval_depth_l1(self, npz_path, stream, global_scale=None):
        """This is from splat-slam, not used in WildGS-SLAM
        """
        # Compute Depth L1 error
        depth_l1_list = []
        depth_l1_list_max_4m = []
        mask_list = []

        # load from disk
        offline_video = dict(np.load(npz_path))
        video_timestamps = offline_video['timestamps']

        for i in range(video_timestamps.shape[0]):
            timestamp = int(video_timestamps[i])
            mask = self.valid_depth_mask[i]
            if mask.sum() == 0:
                print("WARNING: mask is empty!")
            mask_list.append((mask.sum()/(mask.shape[0]*mask.shape[1])).cpu().numpy())
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            # compute scale and shift for depth
            # load gt depth from stream
            depth_gt = stream[timestamp][2].to(self.device)
            mask = torch.logical_and(depth_gt > 0, mask)
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list.append(depth_l1.cpu().numpy())

            # update process but masking depth_gt > 4
            # compute scale and shift for depth
            mask = torch.logical_and(depth_gt < 4, mask)
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list_max_4m.append(depth_l1.cpu().numpy())

        return np.asarray(depth_l1_list).mean(), np.asarray(depth_l1_list_max_4m).mean(), np.asarray(mask_list).mean()