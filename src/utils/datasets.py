import glob
import os
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov
import src.utils.colmap_read_model as read_model
from scipy.spatial.transform import Rotation

def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y

def load_metric_depth(idx,path):
    # omnidata depth
    mono_depth_path = f"{path}/mono_priors/depths/{idx:05d}.npy"
    mono_depth = np.load(mono_depth_path)
    mono_depth_tensor = torch.from_numpy(mono_depth)
    
    return mono_depth_tensor  

def load_img_feature(idx,path,suffix=''):
    # image features
    feat_path = f"{path}/mono_priors/features/{idx:05d}{suffix}.npy"
    feat = np.load(feat_path)
    feat_tensor = torch.from_numpy(feat)
    
    return feat_tensor  


def get_dataset(cfg, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, device='cuda:0'):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.device = device
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.n_img = -1
        self.depth_paths = None
        self.color_paths = None
        self.poses = None
        self.image_timestamps = None

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig = self.fx, self.fy, self.cx, self.cy
        self.H_out, self.W_out = cfg['cam']['H_out'], cfg['cam']['W_out']
        self.H_edge, self.W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']

        self.H_out_with_edge, self.W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        self.intrinsic = torch.as_tensor([self.fx, self.fy, self.cx, self.cy]).float()
        self.intrinsic[0] *= self.W_out_with_edge / self.W
        self.intrinsic[1] *= self.H_out_with_edge / self.H
        self.intrinsic[2] *= self.W_out_with_edge / self.W
        self.intrinsic[3] *= self.H_out_with_edge / self.H
        self.intrinsic[2] -= self.W_edge
        self.intrinsic[3] -= self.H_edge
        self.fx = self.intrinsic[0].item()
        self.fy = self.intrinsic[1].item()
        self.cx = self.intrinsic[2].item()
        self.cy = self.intrinsic[3].item()

        self.fovx = focal2fov(self.fx, self.W_out)
        self.fovy = focal2fov(self.fy, self.H_out)

        self.W_edge_full = int(math.ceil(self.W_edge*self.W/self.W_out_with_edge))
        self.H_edge_full =  int(math.ceil(self.H_edge*self.H/self.H_out_with_edge))
        self.H_out_full, self.W_out_full = self.H - self.H_edge_full * 2, self.W - self.W_edge_full * 2

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None

        self.input_folder = cfg['data']['input_folder']
        if "ROOT_FOLDER_PLACEHOLDER" in self.input_folder:
            self.input_folder = self.input_folder.replace("ROOT_FOLDER_PLACEHOLDER", cfg['data']['root_folder'])


    def __len__(self):
        return self.n_img

    def depthloader(self, index, depth_paths, depth_scale):
        if depth_paths is None:
            return None
        depth_path = depth_paths[index]
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        elif '.npy' in depth_path:
            depth_data = np.load(depth_path).squeeze()
        else:
            raise TypeError(depth_path)
        depth_data = depth_data.astype(np.float32) / depth_scale

        return depth_data

    def get_color(self,index):
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx_orig, self.cx_orig, self.fy_orig, self.cy_orig
            # undistortion is only applied on color image, not depth!
            color_data_fullsize = cv2.undistort(color_data_fullsize, K, self.distortion)

        color_data = cv2.resize(color_data_fullsize, (self.W_out_with_edge, self.H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            color_data = color_data[:, :, :, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            color_data = color_data[:, :, edge:-edge, :]
        return color_data

    def get_intrinsic(self):
        H_out_with_edge, W_out_with_edge = self.H_out + self.H_edge * 2, self.W_out + self.W_edge * 2
        intrinsic = torch.as_tensor([self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig]).float()
        intrinsic[0] *= W_out_with_edge / self.W
        intrinsic[1] *= H_out_with_edge / self.H
        intrinsic[2] *= W_out_with_edge / self.W
        intrinsic[3] *= H_out_with_edge / self.H   
        if self.W_edge > 0:
            intrinsic[2] -= self.W_edge
        if self.H_edge > 0:
            intrinsic[3] -= self.H_edge   
        return intrinsic 
    
    def get_intrinsic_full_resol(self):
        intrinsic = torch.as_tensor([self.fx_orig, self.fy_orig, self.cx_orig, self.cy_orig]).float()
        if self.W_edge > 0:
            intrinsic[2] -= self.W_edge_full
        if self.H_edge > 0:
            intrinsic[3] -= self.H_edge_full
        return intrinsic 
    
    def get_color_full_resol(self,index):
        # not used now
        color_path = self.color_paths[index]
        color_data_fullsize = cv2.imread(color_path)
        if self.distortion is not None:
            K = np.eye(3)
            K[0, 0], K[0, 2], K[1, 1], K[1, 2] = self.fx_orig, self.cx_orig, self.fy_orig, self.cy_orig
            # undistortion is only applied on color image, not depth!
            color_data_fullsize = cv2.undistort(color_data_fullsize, K, self.distortion)

        color_data_fullsize = torch.from_numpy(color_data_fullsize).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        color_data_fullsize = color_data_fullsize.unsqueeze(dim=0)  # [1, 3, h, w]

        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge_full > 0:
            edge = self.W_edge_full
            color_data_fullsize = color_data_fullsize[:, :, :, edge:-edge]

        if self.H_edge_full > 0:
            edge = self.H_edge_full
            color_data_fullsize = color_data_fullsize[:, :, edge:-edge, :]
        return color_data_fullsize


    def __getitem__(self, index):
        color_data = self.get_color(index)

        depth_data_fullsize = self.depthloader(index,self.depth_paths,self.png_depth_scale)
        if depth_data_fullsize is not None:
            depth_data_fullsize = torch.from_numpy(depth_data_fullsize).float()
            outsize = (self.H_out_with_edge, self.W_out_with_edge)
            depth_data = F.interpolate(
                depth_data_fullsize[None, None], outsize, mode='nearest')[0, 0]
        else:
            depth_data = torch.zeros(color_data.shape[-2:])


        # crop image edge, there are invalid value on the edge of the color image
        if self.W_edge > 0:
            edge = self.W_edge
            depth_data = depth_data[:, edge:-edge]

        if self.H_edge > 0:
            edge = self.H_edge
            depth_data = depth_data[edge:-edge, :]

        if self.poses is not None:
            pose = torch.from_numpy(self.poses[index]).float() #torch.from_numpy(np.linalg.inv(self.poses[0]) @ self.poses[index]).float()
        else:
            pose = None

        return index, color_data, depth_data, pose

    def save_gt_poses(self, path, poses):
        # convert rotation matrix to quaternions, save to txt file
        idx = 0
        with open(path, 'w') as f:
            for pose in poses:
                quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
                translation = pose[:3, 3]
                associated_img_path = self.color_paths[idx]
                # remove the extension only .png
                timestamp = float(os.path.basename(associated_img_path)[:-4])
                f.write(f"{timestamp} {translation[0]:.6f} {translation[1]:.6f} {translation[2]:.6f} {quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}\n")
                idx += 1
        print("INFO: GT poses saved to {}".format(path))


class Replica(BaseDataset):
    """This is from splat-slam, never test it (todo)"""
    def __init__(self, cfg, device='cuda:0'):
        super(Replica, self).__init__(cfg, device)
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        if max_frames < 0:
            max_frames = len(self.color_paths)
        self.n_img = len(self.color_paths)

        self.load_poses(f'{self.input_folder}/traj.txt')
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]

        self.w2c_first_pose = np.linalg.inv(self.poses[0])

        self.n_img = len(self.color_paths)


    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            self.poses.append(c2w)


class ScanNet(BaseDataset):
    """This is from splat-slam, never test it (todo)"""
    def __init__(self, cfg, device='cuda:0'):
        super(ScanNet, self).__init__(cfg, device)
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        if max_frames < 0:
            max_frames = len(self.color_paths)
        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]

        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.poses = self.poses[:max_frames][::stride]

        self.n_img = len(self.color_paths)
        print("INFO: {} images got!".format(self.n_img))

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, device)
        # frame_rate is set to be 32 in MonoGS, we make it to 60 to avoid less frame dropped
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=60, pose_correct_bonn = cfg['dataset']=='bonn_dynamic')
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = len(self.color_paths)

        self.color_paths = self.color_paths[:max_frames][::stride]
        self.depth_paths = self.depth_paths[:max_frames][::stride]
        self.poses = self.poses[:max_frames][::stride]

        self.n_img = len(self.color_paths)

        if cfg['save_gt_poses']:
            output_folder = cfg["data"]["output"] + "/" + cfg["scene"]
            self.save_gt_poses(os.path.join(output_folder, 'gt_poses.txt'), self.poses)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.str_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1, pose_correct_bonn=False):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=0)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            # timestamp tx ty tz qx qy qz qw
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])

            if pose_correct_bonn:
                c2w = self.correct_gt_pose_bonn(c2w)

            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w

            poses += [c2w]

        self.w2c_first_pose = inv_pose

        return images, depths, poses
    
    def correct_gt_pose_bonn(self, T):
        """Specific operation for Bonn dynamic dataset"""
        Tm = np.array([[1.0157, 0.1828, -0.2389, 0.0113],
               [0.0009, -0.8431, -0.6413, -0.0098],
               [-0.3009, 0.6147, -0.8085, 0.0111],
               [0, 0, 0, 1]])
        T_ROS = np.zeros((4,4))
        T_ROS[0,0] = -1
        T_ROS[1,2] = 1
        T_ROS[2,1] = 1
        T_ROS[3,3] = 1

        return T_ROS.T @ T @ T_ROS @ Tm

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
   
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose
    
    def save_gt_poses(self, path, poses):
        # convert rotation matrix to quaternions, save to txt file
        idx = 0
        with open(path, 'w') as f:
            for pose in poses:
                quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
                translation = pose[:3, 3]
                associated_img_path = self.color_paths[idx]
                # remove the extension only .png
                timestamp = float(os.path.basename(associated_img_path)[:-4])
                f.write(f"{timestamp} {translation[0]:.6f} {translation[1]:.6f} {translation[2]:.6f} {quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}\n")
                idx += 1
        print("INFO: GT poses saved to {}".format(path))

class RGB_NoPose(BaseDataset):
    def __init__(self, cfg, device='cuda:0'
                 ):
        super(RGB_NoPose, self).__init__(cfg, device)
        # list all png or jpg files in the input_folder
        if cfg['dataset'] == 'droidw':
            self.color_paths = sorted(glob.glob(f'{self.input_folder}/images_anonymized/*.jpg'))
        else:
            self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/rgb/frame*.png'))
            # YouTube dataset is in jpg format
            self.color_paths += sorted(glob.glob(f'{self.input_folder}/frame*.jpg'))
        self.depth_paths = None
        self.poses = None

        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = len(self.color_paths)

        self.color_paths = self.color_paths[:max_frames][::stride]
        if cfg['dataset'] == 'droidw':
            # save timestamp of rgb image to a txt file
            output_folder = cfg["data"]["output"] + "/" + cfg["scene"]
            with open(os.path.join(output_folder, 'timestamps.txt'), 'w') as f:
                for color_path in self.color_paths:
                    timestamp = float(os.path.basename(color_path)[:-4])
                    f.write(f"{timestamp}\n")
        self.n_img = len(self.color_paths)

        print("INFO: {} images got!".format(self.n_img))

class Dycheck(BaseDataset):
    """This is from splat-slam, never test it (todo)"""
    def __init__(self, cfg, device='cuda:0'):
        super(Dycheck, self).__init__(cfg, device)
        stride = cfg['stride']
        max_frames = cfg['max_frames']
        color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'dense/images', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth/1x', '*.npy')), key=lambda x: int(os.path.basename(x)[:-4]))
        if max_frames < 0:
            max_frames = len(color_paths)
        self.color_paths = color_paths[:max_frames][::stride]
        self.depth_paths = depth_paths[:max_frames][::stride]

        self.load_poses(os.path.join(self.input_folder, 'dense'))
        self.poses = self.poses[:max_frames][::stride]

        assert len(self.color_paths) == self.poses.shape[0]
        assert len(self.color_paths) == len(self.depth_paths)

        self.n_img = len(self.color_paths)
        print("INFO: {} images got!".format(self.n_img))

        if cfg['save_gt_poses']:
            output_folder = cfg["data"]["output"] + "/" + cfg["scene"]
            self.save_gt_poses(os.path.join(output_folder, 'gt_poses.txt'), self.poses)

    def save_gt_poses(self, path, poses):
        # convert rotation matrix to quaternions, save to txt file
        idx = 0
        with open(path, 'w') as f:
            for pose in poses:
                quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
                translation = pose[:3, 3]
                associated_img_path = self.color_paths[idx]
                # remove the extension only .png
                timestamp = float(os.path.basename(associated_img_path)[:-4])
                f.write(f"{timestamp} {translation[0]:.6f} {translation[1]:.6f} {translation[2]:.6f} {quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}\n")
                idx += 1
        print("INFO: GT poses saved to {}".format(path))

    def load_poses(self, path):
        gt_cam2w = self.load_colmap_data(path)

        # normalize the poses
        full_t = np.dot(np.linalg.inv(gt_cam2w[-1]), gt_cam2w[0])
        normalize_scale = np.linalg.norm(full_t[:3, 3]) + 1e-8
        gt_cam2w[:, :3, 3] /= normalize_scale
        self.poses = gt_cam2w

    def load_colmap_data(self, realdir):
        """Load colmap data."""
        camerasfile = os.path.join(realdir, "sparse/cameras.bin")
        camdata = read_model.read_cameras_binary(camerasfile)

        list_of_keys = list(camdata.keys())
        cam = camdata[list_of_keys[0]]
        print("Cameras", len(cam))

        imagesfile = os.path.join(realdir, "sparse/images.bin")
        imdata = read_model.read_images_binary(imagesfile)

        w2c_mats = []
        bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])

        names = [imdata[k].name for k in imdata]
        img_keys = [k for k in imdata]

        print("Images #", len(names))
        perm = np.argsort(names)

        points3dfile = os.path.join(realdir, "sparse/points3D.bin")
        pts3d = read_model.read_points3d_binary(points3dfile)

        # extract point 3D xyz
        point_cloud = []
        for key in pts3d:
            point_cloud.append(pts3d[key].xyz)

        upper_bound = 100000

        if upper_bound < len(img_keys):
            print("Only keeping " + str(upper_bound) + " images!")

        for i in perm[0 : min(upper_bound, len(img_keys))]:
            im = imdata[img_keys[i]]
            if "2_" in im.name:
                continue

            if "1_" in im.name:
                continue

            # print(im.name)
            R = im.qvec2rotmat()
            t = im.tvec.reshape([3, 1])
            m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            w2c_mats.append(m)

        w2c_mats = np.stack(w2c_mats, 0)
        # bounds_mats = np.stack(bounds_mats, 0)
        c2w_mats = np.linalg.inv(w2c_mats)

        return c2w_mats


dataset_dict = {
    "tumrgbd": TUM_RGBD,
    "bonn_dynamic": TUM_RGBD,
    "youtube": RGB_NoPose,
    "dycheck": Dycheck,
    "droidw": RGB_NoPose,
}
