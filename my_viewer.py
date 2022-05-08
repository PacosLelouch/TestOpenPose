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
from pyrender.constants import GLTF
import OpenGL.GL as GL
import custom_pyviewer
import custom_pyrenderer

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
    
    def __init__(self, faces, cfg : dict, capture_size, viewport_size=None):
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
            baseColorFactor=(1.0, 1.0, 0.9, 1.0),
            doubleSided=True)

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
        
        self.background_node = None
        self.mesh_node = None
        self.camera_node = None
        
        self.viewport_size = viewport_size if viewport_size is not None else (self.img_res, self.img_res)
            
        
        ratio_fx = self.viewport_size[0] / self.img_res#capture_size[0]
        ratio_fy = self.viewport_size[1] / self.img_res#capture_size[1]
#        ratio_fx = self.viewport_size[0] / image.shape[1]
#        ratio_fy = self.viewport_size[1] / image.shape[0]
        
        camera_center = [self.viewport_size[0] * 0.5, self.viewport_size[1] * 0.5]#[image.shape[1] / 2., image.shape[0] / 2.]
        
        self.camera = pyrender.IntrinsicsCamera(fx=self.focal_length * ratio_fx, fy=self.focal_length * ratio_fy,
                                           cx=camera_center[0], cy=camera_center[1])
        
#        self.camera_node = pyrender.Node(camera=self.camera, matrix=np.eye(4))
#        self.scene.add_node(self.camera_node)
#        self.scene.main_camera_node = self.camera_node#TODO
        '''
        Begin Background
        '''
        self.background_image = np.ones((capture_size[0], capture_size[1], 3), dtype=np.float32)
        self.background_sampler = pyrender.Sampler(name='background_sampler',
                                                   magFilter=GLTF.LINEAR,
                                                   minFilter=GLTF.LINEAR)
        
        self.background_texture = pyrender.Texture(name='background_texture', 
                                                   source=self.background_image, 
                                                   source_channels='RGB',
                                                   #width=image.shape[1],
                                                   #height=image.shape[0],
                                                   sampler=self.background_sampler,
                                                   data_format=GL.GL_FLOAT)
        self.background_material = pyrender.Material(emissiveTexture=self.background_texture, 
                                                                      emissiveFactor=np.array([0.9, 0.1, 1.0]),
                                                                      name='background_material',
                                                                      doubleSided=True)
#        self.background_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=np.array([0.0, 0.0, 0.0, 1.0]),
#                                                                      emissiveTexture=self.background_texture, 
#                                                                      emissiveFactor=np.array([0.9, 0.1, 1.0, 1.0]),
#                                                                      name='background_material',
#                                                                      doubleSided=True)
        
#        half_width = self.viewport_size[0] // 2
#        half_height = self.viewport_size[1] // 2
#        background_z = -camera_pose[3, 2]
#        background_positions = np.array([[-half_width, -half_height, -background_z],
#                                         [half_width, -half_height, -background_z],
#                                         [half_width, half_height, -background_z],
#                                         [-half_width, half_height, -background_z]])
        proj = self.camera.get_projection_matrix(self.viewport_size[0], self.viewport_size[1])
        proj_z = 0.9
        proj_x = 5.0
        proj_y = 5.0
        background_positions_4 = np.array([[-proj_x, -proj_y, proj_z, 1.0],
                                         [proj_x, -proj_y, proj_z, 1.0],
                                         [proj_x, proj_y, proj_z, 1.0],
                                         [-proj_x, proj_y, proj_z, 1.0]]) @ np.linalg.inv(proj).T
        background_positions = background_positions_4[:, :3] / background_positions_4[:, 3].reshape((-1, 1))
        background_texcoords = np.array([[0.0, 0.0],
                                         [1.0, 0.0],
                                         [1.0, 1.0],
                                         [0.0, 1.0]])
        print('unprojected coords =', background_positions)
        
#        proj_coord = background_positions_4 @ proj.T
#        print('projected coords =', proj_coord)
        
        self.background_primitives = [pyrender.Primitive(positions=background_positions, 
                                                         texcoord_0=background_texcoords,
                                                         texcoord_1=background_texcoords,
                                                         indices=np.array([[0, 2, 1], [0, 3, 2]]),
                                                         material=self.background_material)]
                                                         #material=self.mesh_material)]
        self.background_mesh = pyrender.Mesh(self.background_primitives, name='background_mesh')
        if self.background_node:
            scene.remove_node(self.background_node)
        self.background_node = pyrender.Node(mesh=self.background_mesh, name='background')
        scene.add_node(self.background_node)
        
        '''
        End Background
        '''
        
        viewer_flags = dict(show_world_axis=True, show_mesh_axis=True)
        
        self.my_renderer = custom_pyviewer.CustomViewer(scene=scene, 
                                           viewport_size=(self.viewport_size[0], self.viewport_size[1]),
                                           viewer_flags=viewer_flags,
                                           run_in_thread=True,
                                           point_size=1.0)
        self.camera_node = self.scene.main_camera_node
        
        self.scene.remove_node(self.camera_node)
        self.camera_node.camera = self.camera
        self.scene.add_node(self.camera_node)
        self.my_renderer._camera_node = self.camera_node
    
    def __call__(self,
                 vertices: np.array,
                 camera_translation: np.array,
                 image: np.array,
                 full_frame: bool=False,
                 imgname: Optional[str]=None):
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
#        else:
#            # Change the color of the source image.
#            #image = image.clone() * torch.tensor(self.img_std, device=image.device).reshape(3,1,1)
#            #image = image + torch.tensor(self.img_mean, device=image.device).reshape(3,1,1)
#            image = image.permute(1, 2, 0).cpu().numpy()
            
            
        image = cv2.resize(image, self.viewport_size, interpolation=cv2.INTER_AREA)
        
        self.my_renderer.set_caption(self.my_renderer.viewer_flags['window_title'])
        self.my_renderer.render_lock.acquire()
        
        if camera_translation is not None:
            camera_translation[0] *= -1.
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_translation
            
            self.camera_node.matrix = camera_pose
            self.my_renderer._trackball._n_pose = camera_pose
#            if self.camera_node:
#                scene.remove_node(self.camera_node)
#            self.camera_node = pyrender.Node(camera=self.camera, matrix=camera_pose)
#            scene.add_node(self.camera_node)
        else:
            camera_pose = self.camera_node.matrix
        
        '''
        Begin Background
        '''
        
        self.background_image = (image * (1.0 / 255.0)).astype(np.float32)
        #print(self.background_image, self.background_image.shape, self.background_image.dtype)
        
        self.background_texture.source = self.background_image
        
        proj2model = np.eye(4)#camera_pose# @ np.linalg.inv(proj)
#        print('view =', np.linalg.inv(camera_pose))
##        print('view.inv =', camera_pose)
#        print('proj =', proj)
#        print('proj.inv =', np.linalg.inv(proj))
        print('proj2model =', proj2model)
        
        self.background_node.matrix = proj2model
        
        '''
        End Background
        '''
        
            
        '''
        Begin Body
        '''
        #print('mesh.vert[0] =', vertices[0])
        
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
        
        '''
        End Body
        '''
        
        self.my_renderer.render_lock.release()

#        color, rend_depth = self.my_renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
#        color = color.astype(np.float32) / 255.0
#        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
#        output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
#
##        output_img = output_img.astype(np.float32)
##        renderer.delete()
#        return output_img
        return None
        
    def release(self):
        if self.my_renderer.is_active:
            self.my_renderer.close_external()
            while self.my_renderer.is_active:
                pass