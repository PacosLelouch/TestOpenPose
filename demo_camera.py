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

from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.models import ProHMR
from prohmr.optimization import KeypointFitting
from prohmr.utils import recursive_to
from prohmr.datasets import OpenPoseDataset
from prohmr.utils.renderer import Renderer

"""
Start OpenPose Settings
"""'''
try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/openpose/python/openpose/Release')
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/openpose/x64/Release;' +  dir_path + '/openpose/bin;'
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
except Exception as e:
    print(e)
    sys.exit(-1)
    

    '''
"""
End OpenPose Settings
"""

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
dataset = OpenPoseDataset(model_cfg, img_folder=args.img_folder, keypoint_folder=args.keypoint_folder, max_people_per_image=1)
#
## Setup a dataloader with batch_size = 1 (Process images sequentially)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False)

# Setup the renderer
renderer = Renderer(model_cfg, faces=model.smpl.faces)

os.chdir('../../')

if not os.path.exists(args.out_folder):
    os.makedirs(args.out_folder)

# Go over each image in the dataset
#for i, batch in enumerate(tqdm(dataloader)):
cap = cv2.VideoCapture(0)
while cap.isOpened():
    if cv2.waitKey(1) & 0Xff == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    ret, frame = cap.read()
    #print(ret, frame)
    
    frame1 = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame2 = np.array([frame1[:,:,2], frame1[:,:,1], frame1[:,:,0]])
    
    cv2.imshow('frame1', frame1)
    #continue#TEST
    '''missing: batch['keypoints_2d']'''
    
#    batch = None
#    for i, batch0 in enumerate(tqdm(dataloader)):
#        batch = batch0
#        break
    
    batch = { 'has_smpl_params':{} }
    batch['img'] = torch.Tensor(np.array([frame2]))
    
    #batch['has_smpl_params']['body_pose'] = 0
    #print('batch', batch)

    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)

    batch_size = batch['img'].shape[0]
    img_fn = 'video_capture'
    if not args.run_fitting:
        for n in range(batch_size):
            #img_fn, _ = os.path.splitext(os.path.split(batch['imgname'][n])[1])
            regression_img = renderer(out['pred_vertices'][n, 0].detach().cpu().numpy(),
                                      out['pred_cam_t'][n, 0].detach().cpu().numpy(),
                                      batch['img'][n])
            cv2.imshow('render', regression_img[:, :, ::-1])
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
            cv2.imshow('render', regression_img[:, :, ::-1])
            #cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_fitting.{args.out_format}'), 255*fitting_img[:, :, ::-1])
