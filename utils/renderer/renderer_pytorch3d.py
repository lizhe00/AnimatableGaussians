from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.renderer.mesh.shader import ShaderBase, HardDepthShader, HardPhongShader
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
import torch
import numpy as np
import cv2 as cv


class VertexAtrriShader(ShaderBase):
    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)

        images = hard_rgb_blend(texels, fragments, blend_params)
        return images


class Renderer:
    def __init__(self, img_w: int, img_h: int, mvp = None, shader_name = 'vertex_attribute', bg_color = (0, 0, 0), win_name = None, device = 'cuda'):
        self.img_w = img_w
        self.img_h = img_h
        self.device = device
        raster_settings = RasterizationSettings(
            image_size = (img_h, img_w),
            blur_radius = 0.0,
            faces_per_pixel = 1,
            bin_size = None,
            max_faces_per_bin = 50000
        )

        self.shader_name = shader_name
        blend_params = BlendParams(background_color = bg_color)
        if shader_name == 'vertex_attribute':
            shader = VertexAtrriShader(device = device, blend_params = blend_params)
        elif shader_name == 'position':
            shader = VertexAtrriShader(device = device, blend_params = blend_params)
        elif shader_name == 'phong_geometry':
            shader = HardPhongShader(device = device, blend_params = blend_params)
        else:
            raise ValueError('Invalid shader_name')
        self.renderer = MeshRenderer(
            rasterizer = MeshRasterizer(
                cameras = None,
                raster_settings = raster_settings
            ),
            shader = shader
        )

        self.mesh = None

    def set_camera(self, extr, intr = None):
        affine_mat = np.identity(4, np.float32)
        affine_mat[0, 0] = -1
        affine_mat[1, 1] = -1
        extr = affine_mat @ extr
        extr[:3, :3] = np.linalg.inv(extr[:3, :3])
        extr = torch.from_numpy(extr).to(torch.float32).to(self.device)
        if intr is None:  # assume orthographic projection
            cameras = OrthographicCameras(
                focal_length = ((self.img_w / 2., self.img_h / 2.),),
                principal_point = ((self.img_w / 2., self.img_h / 2.),),
                R = extr[:3, :3].unsqueeze(0),
                T = extr[:3, 3].unsqueeze(0),
                in_ndc = False,
                device = self.device,
                image_size = ((self.img_h, self.img_w),)
            )
        else:
            intr = torch.from_numpy(intr).to(torch.float32).to(self.device)
            cameras = PerspectiveCameras(((intr[0, 0], intr[1, 1]),),
                                         ((intr[0, 2], intr[1, 2]),),
                                         extr[:3, :3].unsqueeze(0),
                                         extr[:3, 3].unsqueeze(0),
                                         in_ndc = False,
                                         device = self.device,
                                         image_size = ((self.img_h, self.img_w),))
        self.renderer.rasterizer.cameras = cameras

    def set_model(self, vertices, vertex_attributes = None):
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices)
        if vertex_attributes is not None:
            if isinstance(vertex_attributes, np.ndarray):
                vertex_attributes = torch.from_numpy(vertex_attributes)
            vertex_attributes = vertex_attributes.to(torch.float32).to(self.device)
        vertices = vertices.to(torch.float32).to(self.device)
        faces = torch.arange(0, vertices.shape[0], dtype = torch.int64).to(self.device).reshape(-1, 3)

        if self.shader_name == 'vertex_attribute':
            textures = TexturesVertex([vertex_attributes])
        elif self.shader_name == 'position':
            textures = TexturesVertex([vertices])
        else:
            textures = TexturesVertex([torch.ones_like(vertices)])

        self.mesh = Meshes([vertices], [faces], textures = textures)

    def render(self):
        img = self.renderer(self.mesh, cameras = self.renderer.rasterizer.cameras)
        return img[0].cpu().numpy()


if __name__ == '__main__':
    import trimesh
    import json
    import time

    """ test perspective projection """
    # cam_data = json.load(open('F:/Pose/thuman4/calibration_00.json', 'r'))
    # cam_names = list(cam_data.keys())
    # view_num = len(cam_names)
    # extr_mats = []
    # for view_idx in range(view_num):
    #     extr_mat = np.identity(4, np.float32)
    #     extr_mat[:3, :3] = np.array(cam_data[cam_names[view_idx]]['R'], np.float32).reshape(3, 3)
    #     extr_mat[:3, 3] = np.array(cam_data[cam_names[view_idx]]['T'], np.float32)
    #     extr_mats.append(extr_mat)
    # intr_mats = [np.array(cam_data[cam_names[view_idx]]['K'], np.float32).reshape(3, 3) for view_idx in range(view_num)]
    # img_heights = [cam_data[cam_names[view_idx]]['imgSize'][1] for view_idx in range(view_num)]
    # img_widths = [cam_data[cam_names[view_idx]]['imgSize'][0] for view_idx in range(view_num)]
    #
    # mesh = trimesh.load('E:\\ProjectCode\\AvatarHD\\test_results\\zzr_avatarrex\\posevocab_wHand_diffVolume_wRootfind\\thuman00_cam18_epoch120\\live_geometry/2141.ply', process = False)
    #
    # renderer = Renderer(img_widths[0], img_heights[0], shader_name = 'vertex_attribute')
    # # renderer = Renderer(img_widths[0], img_heights[0], shader_name = 'phong_geometry')
    # vertices = mesh.vertices[mesh.faces.reshape(-1)]
    # normals = mesh.vertex_normals[mesh.faces.reshape(-1)]
    # renderer.set_model(vertices, normals)
    #
    # for view_idx in range(view_num):
    #     time_0 = time.time()
    #     proj_mat = gl_perspective_projection_matrix(
    #         intr_mats[view_idx][0, 0],
    #         intr_mats[view_idx][1, 1],
    #         intr_mats[view_idx][0, 2],
    #         intr_mats[view_idx][1, 2],
    #         renderer.img_w,
    #         renderer.img_h
    #     )
    #     mvp = proj_mat @ extr_mats[view_idx]
    #
    #     renderer.set_camera(extr_mats[view_idx], intr_mats[view_idx])
    #
    #     img = renderer.render()
    #     print('Render cost %f secs' % (time.time() - time_0))
    #
    #     cv.imshow('img', img)
    #     cv.waitKey(0)

    """ test orthographic projection """
    mesh = trimesh.load('../debug/cano_smpl.obj', process = False)
    renderer = Renderer(1024, 1024, shader_name = 'vertex_attribute')

    cano_center = 0.5 * (mesh.vertices.max(0) + mesh.vertices.min(0))
    front_mv = np.identity(4, np.float32)
    front_mv[:3, 3] = -cano_center + np.array([0, 0, -10], np.float32)
    front_mv[1:3] *= -1  # gl2real

    vertices = mesh.vertices[mesh.faces.reshape(-1)].astype(np.float32)
    normals = mesh.vertex_normals[mesh.faces.reshape(-1)].astype(np.float32)
    renderer.set_camera(front_mv)
    renderer.set_model(vertices, normals)
    img = renderer.render()
    cv.imshow('img', img)
    cv.waitKey(0)
