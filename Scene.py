from VectorMath import *
from Presentation import *
from Buffer import *
from RenderTypes import *
from AssetManager import *
from MatrixMath import *
from Renderer import *

from typing import NamedTuple


class Camera:
    def __init__(self, pos: Vec3, target: Vec3, fov: float, ar: float, near_plane: float, far_plane: float):
        self.pos = pos
        self.target = target
        self.fov = fov
        self.ar = ar
        self.near_plane = near_plane
        self.far_plane = far_plane

    def get_view_matrix(self) -> Mat4:
        return make_lookat_matrix(self.pos, self.target, Vec3(0, 1, 0))

    def get_projection_matrix(self) -> Mat4:
        return make_projection_matrix(
            math.radians(self.fov/2), self.ar, self.near_plane, self.far_plane)


class Scene:
    def __init__(self, viewport: Viewport):
        n_samples_per_axis: float = 2

        color_attachment = Buffer([Vec4(0.1, 0.1, 0.1, 1.0) for x in range(viewport.width * viewport.height * (n_samples_per_axis ** 2))],
                                  viewport.width, viewport.height, n_samples_per_axis)

        depth_attachment = Buffer([float("inf") for x in range(viewport.width * viewport.height * (n_samples_per_axis ** 2))],
                                  viewport.width, viewport.height, n_samples_per_axis)

        self.viewport: Viewport = viewport
        self.asset_manager: AssetManager = AssetManager()
        self.framebuffer: Framebuffer = Framebuffer(
            color_attachment, depth_attachment)

        self.camera = Camera(Vec3(0, 0, 0), Vec3(0, 0, 1),
                             90, viewport.width / viewport.height, 0.001, 100)

        self.models: list[list[Mesh]] = []
        self.model_transforms: list[Transform] = []
        self.point_lights: list[PointLight] = []

        setup_turtle(viewport.width, viewport.height)

    def add_model(self, path: str, transform: Transform):
        self.models.append(self.asset_manager.load_model(path))
        self.model_transforms.append(transform)

    def add_point_light(self, light: PointLight):
        self.point_lights.append(light)

    def render(self):
        view_matrix: Mat4 = self.camera.get_view_matrix()
        projection_matrix: Mat4 = self.camera.get_projection_matrix()

        for (model, transform) in zip(self.models, self.model_transforms):
            model_matrix: Mat4 = make_model_matrix(transform)
            normal_matrix: Mat4 = make_normal_matrix(model_matrix)
            vertex_uniforms = VertexUniforms(
                model_matrix=model_matrix, normal_matrix=normal_matrix,
                view_matrix=view_matrix, projection_matrix=projection_matrix)
            for mesh in model:
                material: Material = mesh.material
                magenta_light: PointLight = PointLight(pos=Vec3(
                    0.0, 0.25, 4.0), linear_att=0.22, quadratic_att=0.20,  color=Vec3(1.0, 0.7, 1.0), specular=Vec3(1.0, 1.0, 1.0))

                uniforms: Uniforms = Uniforms(
                    vertex=vertex_uniforms,
                    fragment=FragmentUniforms(
                        material=material, light=magenta_light))

                draw(self.framebuffer, self.viewport, uniforms,
                     (mesh.positions, mesh.normals, mesh.tex_uvs), 0, mesh.num_vertices)

        resolve_buffer(self.framebuffer.color_attachment)

    def present(self):
        present_backbuffer(self.framebuffer.color_attachment, self.viewport)

    def finish(self):
        turtle.done()
