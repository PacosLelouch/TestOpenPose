import taichi as ti
import tina
import pyrender
import numpy as np
import torch
#from pyrender.constants import DEFAULT_Z_NEAR, DEFAULT_Z_FAR

'''
Based on tina.Scene in raster.py
'''
@ti.data_oriented
class TiRasterScene:
    def __init__(self, res, model_view_size, camera_cfg, **options):
        '''
        :param res: (int | tuple) resolution of screen
        :param options: options for the rasterizers

        Creates a Tina scene, the top level structure to manage everything in your scene.
        '''

        self.engine = tina.Engine(res)
        self.res = self.engine.res
        self.options = options
        self.taa = options.get('taa', False)
        self.ibl = options.get('ibl', False)
        self.ssr = options.get('ssr', False)
        self.ssao = options.get('ssao', False)
        self.fxaa = options.get('fxaa', False)
        self.tonemap = options.get('tonemap', True)
        self.blooming = options.get('blooming', False)
        self.bgcolor = options.get('bgcolor', 0)
        
        '''
        Begin set camera intrinsic
        '''
        self.model_view_size = model_view_size
        self.focal_length = camera_cfg.get('FOCAL_LENGTH', None)
#        self.img_res = camera_cfg.get('IMAGE_SIZE', None)
#        self.img_std = camera_cfg.get('IMAGE_STD', None)
#        self.img_mean = camera_cfg.get('IMAGE_MEAN', None)
        
        ratio_fx = self.res[0] / self.model_view_size[1]
        ratio_fy = self.res[1] / self.model_view_size[0]
        camera_center = (self.res[0] * 0.5, self.res[1] * 0.5)
        
        intrinsics_camera = pyrender.IntrinsicsCamera(fx=self.focal_length * ratio_fx, fy=self.focal_length * ratio_fy,
                                                      cx=camera_center[0], cy=camera_center[1])
        self.proj = intrinsics_camera.get_projection_matrix(self.res[0], self.res[1])
        '''
        End set camera intrinsic
        '''

        if not self.ibl:
            self.lighting = tina.Lighting()
        else:
            skybox = tina.Atomsphere()
            skybox = tina.Skybox(skybox.resolution).cook_from(skybox)
            #skybox = tina.Skybox('assets/skybox.jpg', cubic=True)
            self.lighting = tina.SkyboxLighting(skybox)

        self.image = ti.Vector.field(3, float, shape=self.res)
        self.default_material = tina.Diffuse()
        self.post_shaders = []
        self.pre_shaders = []
        self.materials = []
        self.shaders = {}
        self.objects = {}

        if self.ssr:
            self.mtltab = tina.MaterialTable()

            @ti.materialize_callback
            def init_mtltab():
                self.mtltab.clear_materials()
                for material in self.materials:
                    self.mtltab.add_material(material)

        if self.ssao or self.ssr:
            self.norm_buffer = ti.Vector.field(3, float, self.res)
            self.norm_shader = tina.NormalShader(self.norm_buffer)
            self.pre_shaders.append(self.norm_shader)

        if self.ssr:
            self.mtlid_buffer = ti.field(int, self.res)
            if 'texturing' in options:
                self.coor_buffer = ti.Vector.field(2, float, self.res)
                self.coor_shader = tina.TexcoordShader(self.coor_buffer)
                self.pre_shaders.append(self.coor_shader)
            else:
                self.coor_buffer = ti.Vector.field(2, float, (1, 1))

        if self.ssao:
            self.ssao = tina.SSAO(self.res, self.norm_buffer, taa=self.taa)

        if self.ssr:
            self.ssr = tina.SSR(self.res, self.norm_buffer,
                    self.coor_buffer, self.mtlid_buffer, self.mtltab, taa=self.taa)

        if self.blooming:
            self.blooming = tina.Blooming(self.res)

        self.pp_img = self.image

        if self.tonemap:
            self.tonemap = tina.ToneMapping(self.res)

        if self.fxaa:
            self.fxaa = tina.FXAA(self.res)

        if self.taa:
            self.accum = tina.Accumator(self.res)

        if self.ibl:
            self.background_shader = tina.BackgroundShader(self.image, self.lighting)

        if not self.ibl:
            @ti.materialize_callback
            def add_default_lights():
                #TODO: Change to Raymond light.
                self.lighting.add_light(dir=[1, 2, 3], color=[0.9, 0.9, 0.9])
                self.lighting.set_ambient_light([0.1, 0.1, 0.1])

    def _ensure_material_shader(self, material):
        if material in self.materials:
            return

        shader = tina.Shader(self.image, self.lighting, material)

        base_shaders = [shader]
        if self.ssr:
            mtlid = len(self.materials)
            mtlid_shader = tina.ConstShader(self.mtlid_buffer, mtlid)
            base_shaders.append(mtlid_shader)
        shader = tina.ShaderGroup(self.pre_shaders
                + base_shaders + self.post_shaders)

        self.materials.append(material)
        self.shaders[material] = shader

    def add_object(self, object, material=None, raster=None):
        '''
        :param object: (Mesh | Pars | Voxl) object to add into the scene
        :param material: (Material) specify material for shading the object, self.default_material by default
        :param raster: (Rasterizer) specify the rasterizer for this object, automatically guess if not specified
        '''

        assert object not in self.objects
        if material is None:
            material = self.default_material

        if raster is None:
            if hasattr(object, 'get_nfaces'):
                if hasattr(object, 'get_npolygon') and object.get_npolygon() == 2:
                    if not hasattr(self, 'wireframe_raster'):
                        self.wireframe_raster = tina.WireframeRaster(self.engine, **self.options)
                    raster = self.wireframe_raster
                else:
                    if not hasattr(self, 'triangle_raster'):
                        self.triangle_raster = tina.TriangleRaster(self.engine, **self.options)
                    raster = self.triangle_raster
            elif hasattr(object, 'get_npars'):
                if not hasattr(self, 'particle_raster'):
                    self.particle_raster = tina.ParticleRaster(self.engine, **self.options)
                raster = self.particle_raster
            elif hasattr(object, 'sample_volume'):
                if not hasattr(self, 'volume_raster'):
                    self.volume_raster = tina.VolumeRaster(self.engine, **self.options)
                raster = self.volume_raster
            else:
                raise ValueError(f'cannot determine raster type of object: {object}')

        self._ensure_material_shader(material)

        self.objects[object] = tina.namespace(material=material, raster=raster)

    def init_control(self, gui, center=None, theta=None, phi=None, radius=None,
                     fov=60, is_ortho=False, blendish=True):
        '''
        :param gui: (GUI) the GUI to bind with
        :param center: (3 * [float]) the target (lookat) position
        :param theta: (float) the altitude of camera
        :param phi: (float) the longitude of camera
        :param radius: (float) the distance from camera to target
        :param is_ortho: (bool) whether to use orthogonal mode for camera
        :param blendish: (bool) whether to use blender key bindings
        :param fov: (bool) the initial field of view of the camera
        '''
        pass
#        self.control = tina.Control(gui, fov=fov, is_ortho=is_ortho, blendish=blendish)
#        if center is not None:
#            self.control.center[:] = center
#        self.control.init_rot(theta, phi)
#        if radius is not None:
#            self.control.radius = radius
    
    
    @ti.kernel
    def _fill_color_with_image(self, image_ti : ti.template()):
        for i, j in ti.grouped(self.image):
            P = ti.Vector([ti.min(ti.max((i) * (image_ti.shape[0]) / (self.image.shape[0]), 0.0), (image_ti.shape[0] - 1)), 
                          ti.min(ti.max((j) * (image_ti.shape[1]) / (self.image.shape[1]), 0.0), (image_ti.shape[1] - 1))])
            color_bgr = tina.bilerp(image_ti, P)
            for k in ti.static(range(3)):
                self.image[i, self.image.shape[1] - 1 - j][k] = color_bgr[2 - k]
                #self.image[self.image.shape[0] - 1 - i, self.image.shape[1] - 1 - j][k] = color_bgr[2 - k]
        

    def render(self, background_image=None):
        '''
        Render the image to field self.img
        '''

        if self.taa:
            self.engine.randomize_bias(self.accum.count[None] == 0)

        if background_image is None:
            self.image.fill(self.bgcolor)
        else:
            self._fill_color_with_image(background_image)
            
        self.engine.clear_depth()
        for s in self.pre_shaders:
            s.clear_buffer()
        for s in self.post_shaders:
            s.clear_buffer()

        for object, oinfo in self.objects.items():
            shader = self.shaders[oinfo.material]
            oinfo.raster.set_object(object)
            oinfo.raster.render_occup()
            oinfo.raster.render_color(shader)

        if self.ssao:
            self.ssao.render(self.engine)
            self.ssao.apply(self.image)

        if self.ssr:
            self.ssr.render(self.engine, self.image)
            self.ssr.apply(self.image)

        if hasattr(self, 'background_shader'):
            self.engine.render_background(self.background_shader)

        if self.blooming:
            self.blooming.apply(self.image)
        if self.tonemap:
            self.tonemap.apply(self.image)
        if self.fxaa:
            self.fxaa.apply(self.image)
        if self.taa:
            self.accum.update(self.pp_img)

    @property
    def img(self):
        '''
        The final image to be displayed in GUI
        '''
        return self.accum.img if self.taa else self.pp_img

    
    def get_camera(self, cam_translation_np):
        from tina import lookat, orthogonal, perspective, affine
        
        cam_translation_np[0] *= -1.
        R = np.eye(3)

        view = np.linalg.inv(affine(R, cam_translation_np))
        proj = self.proj

        return view, proj
    
    def input(self, gui, cam_translation_np):
        '''
        :param gui: (GUI) GUI to recieve event from

        Feed inputs from the mouse drag events on GUI to control the camera
        '''
        changed = False
        
        view, proj = self.get_camera(cam_translation_np)
        
        self.engine.set_camera(view, proj)

        if changed:
            self.clear()
        return changed

    def clear(self):
        if hasattr(self, 'accum'):
            self.accum.clear()

    def load_gltf(self, path):
        '''
        :param path: (str | readable-stream) path to the gltf file

        Load the scene from a GLTF format file
        '''

        return tina.readgltf(path).extract(self)

    @ti.kernel
    def _fast_export_image(self, out: ti.ext_arr()):
        for x, y in ti.grouped(self.img):
            base = (y * self.res.x + x) * 3
            r, g, b = self.img[x, y]
            out[base + 0] = r
            out[base + 1] = g
            out[base + 2] = b

    def visualize(self):
        with ti.GUI() as gui:
            while gui.running:
                self.input(gui)
                self.render()
                gui.set_image(self.img)
                gui.show()
