"""TODO"""


#import os
#import sys
#from sys import platform
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
#
#dir_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(dir_path + '/ProHMR/ProHMR')
#os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/ProHMR/ProHMR;'

import numpy as np
import cv2
import torch
import pyrender
import trimesh
from typing import List, Optional
#from prohmr.utils.renderer import Renderer

class MyViewer:
    
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
    
    def __init__(self, faces, cfg : dict, viewport_size=None):
        """
        Wrapper around the pyrender renderer to render SMPL meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        self.cfg = cfg
        self.focal_length = cfg.get('FOCAL_LENGTH', None)#cfg.EXTRA.FOCAL_LENGTH
        self.img_res = cfg.get('IMAGE_SIZE', None)#cfg.MODEL.IMAGE_SIZE
        self.img_std = cfg.get('IMAGE_STD', None)
        self.img_mean = cfg.get('IMAGE_MEAN', None)

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces
        
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


        light_nodes = MyViewer.create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)
        
        self.mesh_material = material
        self.scene = scene
        
        self.mesh_node = None
        self.camera_node = None
        
        self.viewport_size = viewport_size if viewport_size is not None else (self.img_res, self.img_res)
        self.my_renderer = pyrender.Viewer(scene=scene, 
                                           viewport_size=(self.viewport_size[0], self.viewport_size[1]),
                                           point_size=1.0,
                                           run_in_thread=True)
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
            # Change the color of the source image.
            #image = image.clone() * torch.tensor(self.img_std, device=image.device).reshape(3,1,1)
            #image = image + torch.tensor(self.img_mean, device=image.device).reshape(3,1,1)
            image = image.permute(1, 2, 0).cpu().numpy()
            
        
        ratio_fx = self.viewport_size[0] / image.shape[1]
        ratio_fy = self.viewport_size[1] / image.shape[0]
        #ratio_fx = image.shape[1] / self.viewport_size[0]
        #ratio_fy = image.shape[0] / self.viewport_size[1]
            
        image = cv2.resize(image, self.viewport_size, interpolation=cv2.INTER_AREA)
        
        self.my_renderer.render_lock.acquire()
        
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
        
        self.my_renderer.render_lock.release()

#        color, rend_depth = self.my_renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
#        color = color.astype(np.float32) / 255.0
#        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
#        output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
#
##        output_img = output_img.astype(np.float32)
##        renderer.delete()
#        return output_img
        
    def release(self):
        if self.my_renderer.is_active:
            self.my_renderer.close_external()
            while self.my_renderer.is_active:
                pass