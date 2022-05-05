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

import taichi as ti
import tina
from ti_raster_scene import TiRasterScene
from ti_torch_mesh import TiTorchMesh
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

def main():
    
    # Use the GPU if available
    cuda_is_available = torch.cuda.is_available()
    device = torch.device('cuda') if cuda_is_available else torch.device('cpu')
    if args.render:
        if args.run_fitting:
            ti.init(ti.opengl if cuda_is_available else ti.cpu) # Conflict with OpenPose?
        else:
            ti.init(ti.cuda if cuda_is_available else ti.cpu)
    
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
    
    os.chdir('../../')
    
    '''
    Begin setup video capture
    '''
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

    '''
    End setup video capture
    '''
    
    '''
    Begin setup the renderer
    '''
    image_ti = ti.Vector.field(3, float, shape=capture_size)
    
    if args.render:
        camera_cfg = {
            'FOCAL_LENGTH': model_cfg.EXTRA.FOCAL_LENGTH,
            'IMAGE_SIZE': model_cfg.MODEL.IMAGE_SIZE,
            'IMAGE_STD': model_cfg.MODEL.IMAGE_STD,
            'IMAGE_MEAN': model_cfg.MODEL.IMAGE_MEAN,
        }
        scene = TiRasterScene(res=viewport_size, model_view_size=(image_width, image_height), camera_cfg=camera_cfg, smoothing=True)
        
        smpl_faces = model.smpl.faces
        print('smpl_faces.shape =', smpl_faces.shape)
        smpl_num_verts = model.smpl.get_num_verts()
        print('smpl_num_verts =', smpl_num_verts)
        
        mesh_raw = TiTorchMesh(faces=smpl_faces, num_verts=smpl_num_verts)
        mesh_transform = tina.MeshTransform(mesh_raw)
        mesh_smooth = tina.MeshSmoothNormal(mesh_transform, cached=False)
        mesh = mesh_smooth
        
        scene.add_object(mesh)
        
        rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
        mesh_transform.set_transform(rot)
        
    gui = ti.GUI(name='ProHMR Demo', res=viewport_size, show_gui=True, fast_gui=False)
    
    '''
    End setup the renderer
    '''
    
    record_time = time.time()
    while gui.running:
        cur_time = time.time()
        fps = (cur_time - record_time)
        fps = -1.0 if fps == 0.0 else 1.0 / fps
        record_time = cur_time
        
        if args.debug_performance:
            print('\n==================Loop Start===================')
            debug_times = [record_time]
        
        window_title = 'render [fps = %.3f (hz)]'%(fps)
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        image_np = np.swapaxes(frame * (1.0 / 255.0), 0, 1)
#        print('image_np.shape =', image_np.shape)
#        print('image_ti.shape =', image_ti.shape)
#        print('image_ti.n =', image_ti.n)
#        print('image_ti.m =', image_ti.m)
        image_ti.from_numpy(image_np)
        
        frame1 = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
        frame2 = np.array([frame1[:,:,0], frame1[:,:,1], frame1[:,:,2]]) * (1.0 / 255.0)
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
            datum.cvInputData = frame1.copy()
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
#            if args.run_open_pose:
#                window_title_OpenPose = 'OpenPose [fps = %.3f (hz)]'%(fps)
#                cv2.imshow(named_window_name_OpenPose, cv2.resize(datum.cvOutputData, viewport_size))
#                cv2.setWindowTitle(named_window_name_OpenPose, window_title_OpenPose)
            
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
        
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
            
        if args.debug_performance:
            print('-----------------End ProHMR Regression-------------------')
            debug_times.append(time.time())
            print('Elapsed %.3f (s)\n' % (debug_times[-1] - debug_times[-2]))
    
        batch_size = batch['img'].shape[0]
        vertices_torch = None
        cam_translate_torch = None
        if batch['keypoints_2d'] is None or not args.run_fitting:
            if batch['keypoints_2d'] is None and args.run_fitting:
                print('batch[\'keypoints_2d\'] is None!')
            for n in range(batch_size):
                vertices_torch = out['pred_vertices'][n, 0].detach()
                cam_translate_torch = out['pred_cam_t'][n, 0].detach()
                
        if batch['keypoints_2d'] is not None and args.run_fitting:
            if args.debug_performance:
                print('-----------------Begin Keypoint Fitting-------------------')
                debug_times.append(time.time())
                
            opt_out = model.downstream_optimization(regression_output=out,
                                                    batch=batch,
                                                    opt_task=keypoint_fitting,
                                                    use_hips=False,
                                                    full_frame=args.full_frame)
            if args.debug_performance:
                print('-----------------End Keypoint Fitting-------------------')
                debug_times.append(time.time())
                print('Elapsed %.3f (s)\n' % (debug_times[-1] - debug_times[-2]))
            
            for n in range(batch_size):
                vertices_torch = opt_out['vertices'][n].detach()
                cam_translate_torch = opt_out['camera_translation'][n].detach()
        
        '''
        Begin rendering.
        '''
        if args.render:
            if args.debug_performance:
                print('-----------------Begin Rendering-------------------')
                debug_times.append(time.time())
                
            #mesh_raw.set_verts_from_numpy(vertices_torch.cpu().numpy().astype(float))#DEBUG
            mesh_raw.set_verts_from_torch(vertices_torch)
            mesh_smooth.update_normal()
            
            scene.input(gui, cam_translate_torch.cpu().numpy())
            scene.render(image_ti)
            
            gui.set_image(scene.img)
            
            if args.debug_performance:
                print('-----------------End Rendering-------------------')
                debug_times.append(time.time())
                print('Elapsed %.3f (s)\n' % (debug_times[-1] - debug_times[-2]))
        else:
            print(window_title)
        
        gui.text(window_title, (100, 100), font_size=72, color=0xFFFF00)
        gui.show()
        '''
        End rendering.
        '''
        if args.debug_performance:
            print('==================Loop End===================\n')

    '''
    Begin release resources.
    '''
    cap.release()
    '''
    End release resources.
    '''
    
    return True

if __name__ == "__main__":
    main()
    