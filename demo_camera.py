"""
ProHMR demo script.
To run our method you need a folder with images and corresponding OpenPose detections.
These are used to crop the images around the humans and optionally to fit the SMPL model on the detections.

Example usage:
python demo.py --checkpoint=path/to/checkpoint.pt --img_folder=/path/to/images --keypoint_folder=/path/to/json --out_folder=/path/to/output --run_fitting

Running the above will run inference for all images in /path/to/images with corresponding keypoint detections.
The rendered results will be saved to /path/to/output, with the suffix _regression.jpg for the regression (mode) and _fitting.jpg for the fitting.

Please keep in mind that we do not recommend to use `--full_frame` when the image resolution is above 2K because of known issues with the data term of SMPLify.
In these cases you can resize all images such that the maximum image dimension is at most 2K.
"""
import os
import sys
from sys import platform
os.environ['KMP_DUPLICATE_LIB_OK']='True'

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/ProHMR/ProHMR')
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/ProHMR/ProHMR;'

#os.chdir('ProHMR/ProHMR')

import numpy as np
import torch
import argparse
import os
import cv2
from tqdm import tqdm
import pyrender
import trimesh
from typing import List, Optional
import time

from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.models import ProHMR
from prohmr.optimization import KeypointFitting
from prohmr.utils import recursive_to
from prohmr.datasets import OpenPoseDataset
from prohmr.utils.renderer import Renderer

class MyRenderer(Renderer):
    
    def create_raymond_lights() -> List[pyrender.Node]:
        """
        Return raymond light nodes for the scene.
        """
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    
        nodes = []
    
        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)
    
            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
    
            matrix = np.eye(4)
            matrix[:3,:3] = np.c_[x,y,z]
            nodes.append(pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            ))
    
        return nodes
    
    def __init__(self, cfg, faces, viewport_size=None):
        super(MyRenderer, self).__init__(cfg, faces)
        
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))

#        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
#        rot = trimesh.transformations.rotation_matrix(
#            np.radians(180), [1, 0, 0])
#        mesh.apply_transform(rot)
#        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
#        scene.add(mesh, 'mesh')


#        camera_translation[0] *= -1.
#        camera_pose = np.eye(4)
#        camera_pose[:3, 3] = camera_translation
#        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
#        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
#                                           cx=camera_center[0], cy=camera_center[1])
#        scene.add(camera, pose=camera_pose)


        light_nodes = MyRenderer.create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)
        
        self.mesh_material = material
        self.scene = scene
        
        self.mesh_node = None
        self.camera_node = None
        
        self.viewport_size = viewport_size if viewport_size is not None else (self.img_res, self.img_res)
        self.my_renderer = pyrender.OffscreenRenderer(viewport_width=self.viewport_size[0],
                                                      viewport_height=self.viewport_size[1],
                                                      point_size=1.0)
        #self.viewer = pyrender.Viewer()
    
    def __call__(self,
                 vertices: np.array,
                 camera_translation: np.array,
                 image: torch.Tensor,
                 full_frame: bool=False,
                 imgname: Optional[str]=None) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """
        #print('vertices:', vertices)#TEST
        #vertices = 
        scene = self.scene
        
        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        else:
            image = image.clone() * torch.tensor(self.cfg.MODEL.IMAGE_STD, device=image.device).reshape(3,1,1)
            image = image + torch.tensor(self.cfg.MODEL.IMAGE_MEAN, device=image.device).reshape(3,1,1)
            image = image.permute(1, 2, 0).cpu().numpy()
            
        
        ratio_fx = self.viewport_size[0] / image.shape[1]
        ratio_fy = self.viewport_size[1] / image.shape[0]
        #ratio_fx = image.shape[1] / self.viewport_size[0]
        #ratio_fy = image.shape[0] / self.viewport_size[1]
            
        image = cv2.resize(image, self.viewport_size, interpolation=cv2.INTER_AREA)
        
        camera_translation[0] *= -1.
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [self.viewport_size[0] * 0.5, self.viewport_size[1] * 0.5]#[image.shape[1] / 2., image.shape[0] / 2.]
        
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length * ratio_fx, fy=self.focal_length * ratio_fy,
                                           cx=camera_center[0], cy=camera_center[1])
        
        if self.camera_node:
            scene.remove_node(self.camera_node)
        self.camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(self.camera_node)
        
        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=self.mesh_material)
            
        if self.mesh_node:
            scene.remove_node(self.mesh_node)
        self.mesh_node = pyrender.Node(mesh=mesh, name='mesh')
        scene.add_node(self.mesh_node)
        #scene.add(mesh, 'mesh')

#        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
#                                              viewport_height=image.shape[0],
#                                              point_size=1.0)

        color, rend_depth = self.my_renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)

#        output_img = output_img.astype(np.float32)
#        renderer.delete()
        return output_img
    


parser = argparse.ArgumentParser(description='ProHMR demo code')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
parser.add_argument('--img_folder', type=str, required=True, help='Folder with input images')
parser.add_argument('--keypoint_folder', type=str, required=True, help='Folder with corresponding OpenPose detections')
parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
parser.add_argument('--out_format', type=str, default='jpg', choices=['jpg', 'png'], help='Output image format')
parser.add_argument('--run_fitting', dest='run_fitting', action='store_true', default=False, help='If set, run fitting on top of regression')
parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, run fitting in the original image space and not in the crop.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')


args = parser.parse_args()

"""
Start OpenPose Settings
"""

open_pose_running = False

if args.run_fitting:
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print('dir_path =', dir_path) #TEST
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/openpose/build/python/openpose/Release')
                os.environ['OpenPose_root_path'] = dir_path + '/openpose'
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/openpose/build/x64/Release;' +  dir_path + '/openpose/build/bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('openpose/python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e
        
        opparams = dict()
        opparams['model_folder'] = dir_path + '/openpose/models'
        #opWrapper = op.WrapperPython(op.ThreadManagerMode.Synchronous)
        opWrapper = op.WrapperPython()
        opWrapper.configure(opparams)
        #opWrapper.execute()
        opWrapper.start()
        
        #sys.exit(0)
        
        open_pose_running = True
    except Exception as e:
        print(e)
        sys.exit(-1)


"""
End OpenPose Settings
"""

# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


os.chdir('ProHMR/ProHMR/')
if args.model_cfg is None:
    model_cfg = prohmr_config()
else:
    model_cfg = get_config(args.model_cfg)

# Setup model
model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()

if args.run_fitting:
    keypoint_fitting = KeypointFitting(model_cfg)

## Create a dataset on-the-fly
#dataset = OpenPoseDataset(model_cfg, img_folder=args.img_folder, keypoint_folder=args.keypoint_folder, max_people_per_image=1)
##
## Setup a dataloader with batch_size = 1 (Process images sequentially)
#dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False)

# Setup video capture
cap = cv2.VideoCapture(0)
image_width = model_cfg.MODEL.IMAGE_SIZE
image_height = model_cfg.MODEL.IMAGE_SIZE
viewport_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('viewport_size =', viewport_size)

# Setup the renderer
renderer = MyRenderer(model_cfg, faces=model.smpl.faces, viewport_size=viewport_size)

os.chdir('../../')

if not os.path.exists(args.out_folder):
    os.makedirs(args.out_folder)

# Go over each image in the dataset
#for i, batch in enumerate(tqdm(dataloader)):
named_window_name = 'Demo Camera ProHMR'
cv2.namedWindow(named_window_name)
record_time = 0.0
while cap.isOpened():
    if cv2.waitKey(1) & 0Xff == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    
    cur_time = time.time()
    fps = (cur_time - record_time)
    fps = -1.0 if fps == 0.0 else 1.0 / fps
    record_time = cur_time
    
    window_title = 'render [fps = %.3f (hz)]'%(fps)
    
    ret, frame = cap.read()
    #print(ret, frame)
    #print(frame.shape)
    frame1 = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    frame2 = np.array([frame1[:,:,0], frame1[:,:,1], frame1[:,:,2]]) * (1.0 / 255.0)
    #frame2 = np.array([frame1[:,:,2], frame1[:,:,1], frame1[:,:,0]])
        
    img_fn = 'video_capture'
    
    batch = { 'has_smpl_params':{} }
    batch['img'] = torch.Tensor(np.array([frame2]))
    batch['imgname'] = [img_fn]
    if open_pose_running:
        datum = op.Datum()
        datum.cvInputData = frame1
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        #print('cvInputData: ', datum.cvInputData)
        #print('cvOutputData: ', datum.cvOutputData) # 0-255
        #print('outputData: ', datum.outputData)
        #print('poseKeypoints: ', datum.poseKeypoints)
        #print('poseKeypoints.shape: ', datum.poseKeypoints.shape)
        #print('handKeypoints[0].shape: ', datum.handKeypoints[0].shape)
        #print('handKeypoints[1].shape: ', datum.handKeypoints[1].shape)
        #print('faceKeypoints.shape: ', datum.faceKeypoints.shape)
        
        #print(frame1)
        
        #cv2.imshow('datum.cvOutputData', datum.cvOutputData)
        
        pose_keypoints = np.zeros((1, 44, 3))
        pose_keypoints[:, :datum.poseKeypoints.shape[1], :] = datum.poseKeypoints
        #print(pose_keypoints, pose_keypoints.shape)
        batch['keypoints_2d'] = torch.Tensor(pose_keypoints)
        #print(batch['keypoints_2d'])
    
#    batch = None
#    for i, batch0 in enumerate(tqdm(dataloader)):
#        batch = batch0
#        break
    
    #batch['has_smpl_params']['body_pose'] = 0
    #print('batch', batch)

    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)

    batch_size = batch['img'].shape[0]
    if not args.run_fitting:
        for n in range(batch_size):
            #img_fn, _ = os.path.splitext(os.path.split(batch['imgname'][n])[1])
            regression_img = renderer(out['pred_vertices'][n, 0].detach().cpu().numpy(),
                                      out['pred_cam_t'][n, 0].detach().cpu().numpy(),
                                      batch['img'][n])
            image_show = regression_img[:, :, ::-1]#cv2.resize(regression_img[:, :, ::-1], (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow(named_window_name, image_show)
            cv2.setWindowTitle(named_window_name, window_title)
            #cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_regression.{args.out_format}'), 255*regression_img[:, :, ::-1])
    if args.run_fitting:
        opt_out = model.downstream_optimization(regression_output=out,
                                                batch=batch,
                                                opt_task=keypoint_fitting,
                                                use_hips=False,
                                                full_frame=args.full_frame)
        for n in range(batch_size):
            #img_fn, _ = os.path.splitext(os.path.split(batch['imgname'][n])[1])
            fitting_img = renderer(opt_out['vertices'][n].detach().cpu().numpy(),
                                   opt_out['camera_translation'][n].detach().cpu().numpy(),
                                   batch['img'][n], imgname=batch['imgname'][n], full_frame=args.full_frame)
            image_show = fitting_img[:, :, ::-1]#cv2.resize(fitting_img[:, :, ::-1], (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow(named_window_name, image_show)
            cv2.setWindowTitle(named_window_name, window_title)
            #cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_fitting.{args.out_format}'), 255*fitting_img[:, :, ::-1])
