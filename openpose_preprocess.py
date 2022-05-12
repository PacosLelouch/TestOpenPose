import numpy as np
from prohmr.datasets.utils import fliplr_keypoints
from typing import List, Dict, Tuple
import prohmr.datasets.utils as prohmr_utils 
from yacs.config import CfgNode
import torch

class OpenPosePreprocess:
    def __init__(self, 
                 cfg : dict,
                 rescale_factor: float = 1.2):
        body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
        extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
        flip_keypoint_permutation = body_permutation + [25 + i for i in extra_permutation]
        
        self.focal_length = cfg.get('FOCAL_LENGTH', None)
        
        self.img_size = cfg.get('IMAGE_SIZE', None)
        self.mean = 255. * cfg.get('IMAGE_MEAN', np.zeros((3,)))
        self.std = 255. * cfg.get('IMAGE_STD', np.zeros((3,)))
        
        self.dataset_config = cfg.get('DATASET_CONFIG', CfgNode())
        self.num_body_joints = cfg.get('NUM_BODY_JOINTS', 23)
        self.rescale_factor = rescale_factor
        
        self.imgname = ['imgname']
        
        self.flip_keypoint_permutation = flip_keypoint_permutation
        
    def _preprocess(self, 
                   body_keypoints_2D):
        keypoints_n = body_keypoints_2D
        keypoints_valid_n = keypoints_n[keypoints_n[:, 1] > 0, :].copy()
        bbox = [0.0, 0.0, 0.0, 0.0]
        if keypoints_valid_n.shape[0] > 0:
            bbox = [min(keypoints_valid_n[:,0]), min(keypoints_valid_n[:,1]),
                    max(keypoints_valid_n[:,0]), max(keypoints_valid_n[:,1])]
        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        scale = self.rescale_factor * max(bbox[2]-bbox[0], bbox[3]-bbox[1])
        
        body_keypoints = [keypoints_n]
        centers = [center]
        scales = [scale]
        
        self.scale = np.array(scales).astype(np.float32) / 200.0
        self.center = np.array(centers).astype(np.float32)
        body_keypoints_2d = np.array(body_keypoints).astype(np.float32)
        extra_keypoints_2d = np.zeros((len(self.center), 19, 3))
        self.keypoints_2d = np.concatenate((body_keypoints_2d, extra_keypoints_2d), axis=1).astype(np.float32)
        body_keypoints_3d = np.zeros((len(self.center), 25, 4), dtype=np.float32)
        extra_keypoints_3d = np.zeros((len(self.center), 19, 4), dtype=np.float32)
        self.keypoints_3d = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1).astype(np.float32)
        num_pose = 3 * (self.num_body_joints + 1)
        self.body_pose = np.zeros((len(self.imgname), num_pose), dtype=np.float32)
        self.has_body_pose = np.zeros(len(self.imgname), dtype=np.float32)
        self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32)
        self.has_betas = np.zeros(len(self.imgname), dtype=np.float32)
        
    def preprocess_item(self, 
                        item : dict,
                        imgname : str,
                        image : np.array,
                        body_keypoints_2D, 
                        extra_keypoints_2D=None,
                        body_keypoints_3D=None,
                        extra_keypoints_3D=None):
        
        if body_keypoints_2D is None:
            body_keypoints_2D = np.zeros((25, 3))
        
        self._preprocess(body_keypoints_2D)
        '''
        Modify item and return item
        '''
        idx = 0
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d[idx].copy()

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = self.scale[idx]*200
        body_pose = self.body_pose[idx].copy().astype(np.float32)
        betas = self.betas[idx].copy().astype(np.float32)

        has_body_pose = self.has_body_pose[idx].copy()
        has_betas = self.has_betas[idx].copy()

        smpl_params = {'global_orient': body_pose[:3],
                       'body_pose': body_pose[3:],
                       'betas': betas
                      }

        has_smpl_params = {'global_orient': has_body_pose,
                           'body_pose': has_body_pose,
                           'betas': has_betas
                           }

        smpl_params_is_axis_angle = {'global_orient': True,
                                     'body_pose': True,
                                     'betas': False
                                    }

        augm_config = self.dataset_config#self.cfg.DATASETS.CONFIG
        # Crop image and (possibly) perform data augmentation
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = OpenPosePreprocess.get_example(image,
                                                                                                                       center_x, center_y,
                                                                                                                       bbox_size, bbox_size,
                                                                                                                       keypoints_2d, keypoints_3d,
                                                                                                                       smpl_params, has_smpl_params,
                                                                                                                       self.flip_keypoint_permutation,
                                                                                                                       self.img_size, self.img_size,
                                                                                                                       self.mean, self.std, False, augm_config)
        # These are the keypoints in the original image coordinates (before cropping)
        orig_keypoints_2d = self.keypoints_2d[idx].copy()

        item['imgname'] = imgname
        item['img'] = img_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = self.scale[idx] * 200
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['idx'] = idx
        
        def recursive_to_tensor(d):
            for key in d:
                if isinstance(d[key], np.ndarray):
                    d[key] = torch.Tensor([d[key]])
                elif isinstance(d[key], dict):
                    recursive_to_tensor(d[key])
                elif isinstance(d[key], float):
                    d[key] = torch.Tensor([d[key]])
                elif isinstance(d[key], int):
                    d[key] = torch.Tensor([d[key]])
                else:
                    d[key] = [d[key]]
        
        recursive_to_tensor(item)
                    
        return item
    
    def get_example(cvimg: np.array, center_x: float, center_y: float,
                    width: float, height: float,
                    keypoints_2d: np.array, keypoints_3d: np.array,
                    smpl_params: Dict, has_smpl_params: Dict,
                    flip_kp_permutation: List[int],
                    patch_width: int, patch_height: int,
                    mean: np.array, std: np.array,
                    do_augment: bool, augm_config: CfgNode) -> Tuple:
        img_height, img_width, img_channels = cvimg.shape
    
        img_size = np.array([img_height, img_width])
    
        # 2. get augmentation params
        if do_augment:
            scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty = prohmr_utils.do_augmentation(augm_config)
        else:
            scale, rot, do_flip, do_extreme_crop, color_scale, tx, ty = 1.0, 0, False, False, [1.0, 1.0, 1.0], 0., 0.
    
        if do_extreme_crop:
            center_x, center_y, width, height = prohmr_utils.extreme_cropping(center_x, center_y, width, height, keypoints_2d)
        center_x += width * tx
        center_y += height * ty
    
        # Process 3D keypoints
        keypoints_3d = prohmr_utils.keypoint_3d_processing(keypoints_3d, flip_kp_permutation, rot, do_flip)
    
        # 3. generate image patch
        img_patch_cv, trans = prohmr_utils.generate_image_patch(cvimg,
                                                                center_x, center_y,
                                                                width, height,
                                                                patch_width, patch_height,
                                                                do_flip, scale, rot)
    
        image = img_patch_cv.copy()
        image = image[:, :, ::-1]
        #img_patch_cv = image.copy()
        img_patch = prohmr_utils.convert_cvimg_to_tensor(image)
    
    
        smpl_params, has_smpl_params = prohmr_utils.smpl_param_processing(smpl_params, has_smpl_params, rot, do_flip)
    
        # apply normalization
        for n_c in range(img_channels):
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
            if mean is not None and std is not None:
                img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
        if do_flip:
            keypoints_2d = prohmr_utils.fliplr_keypoints(keypoints_2d, img_width, flip_kp_permutation)
    
    
        for n_jt in range(len(keypoints_2d)):
            keypoints_2d[n_jt, 0:2] = prohmr_utils.trans_point2d(keypoints_2d[n_jt, 0:2], trans)
        keypoints_2d[:, :-1] = keypoints_2d[:, :-1] / patch_width - 0.5
    
        return img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size