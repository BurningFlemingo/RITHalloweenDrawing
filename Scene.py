from VectorMath import *
from Presentation import *
from Buffer import *
from RenderTypes import *
from AssetManager import *
from MatrixMath import *
from Renderer import *
from dataclasses import dataclass

from typing import NamedTuple


@dataclass
class Camera:
    pos: Vec3
    target: Vec3
    fov: float
    near_plane: float
    far_plane: float
    
    def __init__(self, pos: Vec3, target: Vec3, fov: float, near_plane: float, far_plane: float):
        self.pos = Vec3(*pos)
        self.target = Vec3(*target)
        self.fov = fov
        self.near_plane = near_plane
        self.far_plane = far_plane




class Scene:
    def __init__(self, viewport: Viewport):
        n_samples_per_axis: float = 2

        color_attachment = Buffer([Vec4(0.1, 0.1, 0.1, 1.0) for x in range(viewport.width * viewport.height * (n_samples_per_axis ** 2))],
                                  viewport.width, viewport.height, n_samples_per_axis, srgb_nonlinear=False)
        resolve_attachment = Buffer([Vec3(0.0, 0.0, 0.0) for x in range(viewport.width * viewport.height)],
                                  viewport.width, viewport.height, 1, srgb_nonlinear=True)

        depth_attachment = Buffer([float("inf") for x in range(viewport.width * viewport.height * (n_samples_per_axis ** 2))],
                                  viewport.width, viewport.height, n_samples_per_axis, srgb_nonlinear=False)

        self.viewport: Viewport = viewport
        self.asset_manager: AssetManager = AssetManager()
        self.framebuffer: Framebuffer = Framebuffer(
            color_attachment, resolve_attachment, depth_attachment)

        self.view_matrix: Mat4 = None
        self.projection_matrix: Mat4 = None
        
        self.models: list[list[Mesh]] = []
        self.model_transforms: list[Transform] = []
        self.point_lights: list[PointLight] = []

        setup_turtle(viewport.width, viewport.height)

    def add_model(self, path: str, transform: Transform):
        self.models.append(self.asset_manager.load_model(path))
        self.model_transforms.append(transform)

    def add_light(self, light: PointLight):
        if (type(light) is PointLight):
            self.point_lights.append(light)

    def set_camera(self, cam: Camera) -> int:
        ar: float = self.viewport.width / self.viewport.height
        
        self.view_matrix = make_lookat_matrix(cam.pos, cam.target, Vec3(0, 1, 0))
        self.projection_matrix = make_projection_matrix(
                cam.fov / 2, 
                ar, 
                cam.near_plane, 
                cam.far_plane
            )
        

    def render(self):
        for (model, transform) in zip(self.models, self.model_transforms):
            model_matrix: Mat4 = make_model_matrix(transform)
            normal_matrix: Mat4 = make_normal_matrix(model_matrix)
            vertex_uniforms = VertexUniforms(
                model_matrix=model_matrix, normal_matrix=normal_matrix,
                view_matrix=self.view_matrix, projection_matrix=self.projection_matrix)
            for mesh in model:
                material: Material = mesh.material

                uniforms: Uniforms = Uniforms(
                    vertex=vertex_uniforms,
                    fragment=FragmentUniforms(
                        material=material, point_lights=self.point_lights))

                draw(self.framebuffer, self.viewport, uniforms,
                     (mesh.positions, mesh.normals, mesh.tex_uvs), 0, mesh.num_vertices)

        resolve_buffer(src=self.framebuffer.color_attachment, target=self.framebuffer.resolve_attachment)

    def present(self):
        present_backbuffer(self.framebuffer.resolve_attachment, self.viewport)

    def finish(self):
        turtle.done()
