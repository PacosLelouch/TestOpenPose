import taichi as ti
import tina
import numpy as np
import torch
#from pyrender.constants import DEFAULT_Z_NEAR, DEFAULT_Z_FAR

@ti.data_oriented
class TiTorchMesh(tina.SimpleMesh):
    # smpl.faces
    # smpl.get_num_verts()
    def __init__(self, faces : np.ndarray, num_verts : int):
        '''
        faces : np.array, size=(m, 3)
        num_verts : int
        '''
        self.maxfaces = faces.shape[0]
        self.maxverts = num_verts
        self.maxcoors = num_verts
        self.maxnorms = num_verts
        self.npolygon = faces.shape[1]

        #self.faces = ti.Matrix.field(3, 3, int, self.maxfaces)
        self.faces = ti.Vector.field(self.npolygon, int, self.maxfaces)
        self.verts = ti.Vector.field(3, float, self.maxverts)
        self.coors = ti.Vector.field(2, float, self.maxcoors)
        self.norms = ti.Vector.field(3, float, self.maxnorms)

        @ti.materialize_callback
        def init_mesh():
            self.faces.from_numpy(faces.astype(int))
#            self.verts.from_numpy(obj['v'])
#            self.coors.from_numpy(obj['vt'])
#            self.norms.from_numpy(obj['vn'])

    def set_verts_from_numpy(self, verts_np : np.ndarray):
        self.verts.from_numpy(verts_np)
        
    def set_verts_from_torch(self, verts_torch : torch.Tensor):
        self.verts.from_torch(verts_torch)

    def get_npolygon(self):
        return self.npolygon

    @ti.func
    def pre_compute(self):
        pass

    def get_max_vert_nindex(self):
        return self.maxverts

    @ti.func
    def get_nfaces(self):
        return self.faces.shape[0]

    @ti.func
    def get_face_vert_indices(self, n):
        i = self.faces[n][0]
        j = self.faces[n][1]
        k = self.faces[n][2]
        return i, j, k

#    @ti.func
#    def _get_face_props(self, prop, index: ti.template(), n):
#        a = prop[self.faces[n][0, index]]
#        b = prop[self.faces[n][1, index]]
#        c = prop[self.faces[n][2, index]]
#        return a, b, c
    @ti.func
    def _get_face_props(self, prop, n):
        a = prop[self.faces[n][0]]
        b = prop[self.faces[n][1]]
        c = prop[self.faces[n][2]]
        return a, b, c

    @ti.func
    def get_face_verts(self, n):
        return self._get_face_props(self.verts, n)

    @ti.func
    def get_face_coors(self, n):
        return self._get_face_props(self.coors, n)

    @ti.func
    def get_face_norms(self, n):
        return self._get_face_props(self.norms, n)