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

from my_viewer import MyViewer
from my_video_capture import MyVideoCapture


parser = argparse.ArgumentParser(description='ProHMR demo code')
parser.add_argument('--checkpoint', type=str, default='data/checkpoint.pt', help='Path to pretrained model checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (prohmr/configs/prohmr.yaml)')
parser.add_argument('--img_folder', type=str, required=True, help='Folder with input images')
parser.add_argument('--keypoint_folder', type=str, required=True, help='Folder with corresponding OpenPose detections')
parser.add_argument('--run_fitting', dest='run_fitting', action='store_true', default=False, help='If set, run fitting on top of regression')
parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, run fitting in the original image space and not in the crop.')
parser.add_argument('--cam', type=str, default="", help='Camera path. Default is 0.')
parser.add_argument('--run_open_pose', dest='run_open_pose', action='store_true', default=False, help='Run open pose for debug.')
parser.add_argument('--screen_width', type=int, default=-1, help='Screen width. Default is -1 for not changing.')
parser.add_argument('--render', type=int, default=1, help='Render result. Default is 1.')
parser.add_argument('--debug_performance', type=int, default=0, help='Debug performance. Default is 0.')


args = parser.parse_args()

"""
Start OpenPose Settings
"""

should_run_open_pose = args.run_fitting or args.run_open_pose
open_pose_running = False

if should_run_open_pose:
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
cam_path = args.cam
cap = MyVideoCapture(0 if cam_path == "" else cam_path)
#if cam_path != "":
#    cap.open(cam_path)
image_width = model_cfg.MODEL.IMAGE_SIZE
image_height = model_cfg.MODEL.IMAGE_SIZE

capture_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
image_ratio = 1.0 if args.screen_width == -1 else args.screen_width / capture_size[0]
viewport_size = (int(capture_size[0] * image_ratio), int(capture_size[1] * image_ratio))
print('viewport_size =', viewport_size)

# Setup the renderer
#print(type(model.smpl.faces), model.smpl.get_num_verts())
renderer = MyViewer(faces=model.smpl.faces, 
                      cfg={
                              'FOCAL_LENGTH': model_cfg.EXTRA.FOCAL_LENGTH,
                              'IMAGE_SIZE': model_cfg.MODEL.IMAGE_SIZE,
                              'IMAGE_STD': model_cfg.MODEL.IMAGE_STD,
                              'IMAGE_MEAN': model_cfg.MODEL.IMAGE_MEAN,
                              },
                      capture_size=capture_size,
                      viewport_size=viewport_size)

os.chdir('../../')

#if not os.path.exists(args.out_folder):
#    os.makedirs(args.out_folder)

# Go over each image in the dataset
#for i, batch in enumerate(tqdm(dataloader)):

record_time = 0.0
while cap.isOpened() and renderer.my_renderer.is_active:
    if cv2.waitKey(1) & 0Xff == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    
    cur_time = time.time()
    fps = (cur_time - record_time)
    fps = -1.0 if fps == 0.0 else 1.0 / fps
    record_time = cur_time
    
    if args.debug_performance:
        print('\n==================Loop Start===================')
        debug_times = [record_time]
    
    window_title = 'render [fps = %.3f (hz)]'%(fps)
    
    renderer.my_renderer.viewer_flags['window_title'] = window_title
    print('window_title:', window_title)
    
    ret, frame = cap.read()
    if not ret:
        continue
    #print(ret, frame)
    #print(frame.shape)
    frame1 = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    frame2 = np.array([frame1[:,:,0], frame1[:,:,1], frame1[:,:,2]]) * (1.0 / 255.0)
    #frame2 = np.array([frame1[:,:,2], frame1[:,:,1], frame1[:,:,0]])
        
    img_fn = 'video_capture'
    
    batch = { 'has_smpl_params':{} }
    batch['img'] = torch.Tensor(np.array([frame2]))
    batch['imgname'] = [img_fn]
    batch['keypoints_2d'] = None
    if open_pose_running:
        if args.debug_performance:
            print('-----------------Begin Open Pose-------------------')
            debug_times.append(time.time())
            
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
        
        if args.run_open_pose:
            window_title_OpenPose = 'OpenPose [fps = %.3f (hz)]'%(fps)
            
        
        pose_keypoints = np.zeros((1, 44, 3))
        if datum.poseKeypoints is not None:
            pose_keypoints[-1, :datum.poseKeypoints.shape[1], :] = datum.poseKeypoints[-1]
            #print(pose_keypoints, pose_keypoints.shape)
            batch['keypoints_2d'] = torch.Tensor(pose_keypoints)
        #print(batch['keypoints_2d'])
        if args.debug_performance:
            print('-----------------End Open Pose-------------------')
            debug_times.append(time.time())
            print('Elapsed %.3f (s)\n' % (debug_times[-1] - debug_times[-2]))
    
    if args.debug_performance:
        print('-----------------Begin ProHMR Regression-------------------')
        debug_times.append(time.time())
#    batch = None
#    for i, batch0 in enumerate(tqdm(dataloader)):
#        batch = batch0
#        break
    
    #batch['has_smpl_params']['body_pose'] = 0
    #print('batch', batch)

    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)
    
    if args.debug_performance:
        print('-----------------End ProHMR Regression-------------------')
        debug_times.append(time.time())
        print('Elapsed %.3f (s)\n' % (debug_times[-1] - debug_times[-2]))

    batch_size = batch['img'].shape[0]
    image_show = None
    if batch['keypoints_2d'] is None or not args.run_fitting:
        if args.render:
            if args.debug_performance:
                print('-----------------Begin Rendering-------------------')
                debug_times.append(time.time())
            for n in range(batch_size):
                #img_fn, _ = os.path.splitext(os.path.split(batch['imgname'][n])[1])
                regression_img = renderer(out['pred_vertices'][n, 0].detach().cpu().numpy(),
                                          out['pred_cam_t'][n, 0].detach().cpu().numpy(),
                                          frame)
                #image_show = regression_img[:, :, ::-1]#cv2.resize(regression_img[:, :, ::-1], (640, 480), interpolation=cv2.INTER_AREA)
        
    if batch['keypoints_2d'] is not None and args.run_fitting:
    
        if args.debug_performance:
            print('-----------------Begin Keypoint Fitting-------------------')
            debug_times.append(time.time())
            
        opt_out = model.downstream_optimization(regression_output=out,
                                                batch=batch,
                                                opt_task=keypoint_fitting,
                                                use_hips=False,
                                                full_frame=args.full_frame)
        if args.render:
            if args.debug_performance:
                print('-----------------End Keypoint Fitting-------------------')
                debug_times.append(time.time())
                print('Elapsed %.3f (s)\n' % (debug_times[-1] - debug_times[-2]))
                    
            if args.debug_performance:
                print('-----------------Begin Rendering-------------------')
                debug_times.append(time.time())
            for n in range(batch_size):
                #img_fn, _ = os.path.splitext(os.path.split(batch['imgname'][n])[1])
                fitting_img = renderer(opt_out['vertices'][n].detach().cpu().numpy(),
                                       opt_out['camera_translation'][n].detach().cpu().numpy(),
                                       frame, imgname=batch['imgname'][n], full_frame=args.full_frame)
                #image_show = fitting_img[:, :, ::-1]#cv2.resize(fitting_img[:, :, ::-1], (640, 480), interpolation=cv2.INTER_AREA)
           
#    if args.render: 
#        cv2.imshow(named_window_name, image_show)
#        cv2.setWindowTitle(named_window_name, window_title)
#        if args.debug_performance:
#            print('-----------------End Rendering-------------------')
#            debug_times.append(time.time())
#            print('Elapsed %.3f (s)\n' % (debug_times[-1] - debug_times[-2]))
                
renderer.release()
