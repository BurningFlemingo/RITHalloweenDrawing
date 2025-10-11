from typing import NamedTuple

from VectorMath import *
from Presentation import *
from Buffer import *
from RenderTypes import *
from AssetManager import *
from MatrixMath import *
from Renderer import *
from dataclasses import dataclass


from shaders.PhongLighting import *
from shaders.ShadowPass import *


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

        shadow_viewport: Viewport = Viewport(width=viewport.width * 4, height=viewport.height * 4)
        
        shadow_map = Buffer([float("inf") for x in range(shadow_viewport.width * shadow_viewport.height)],
            shadow_viewport.width, shadow_viewport.height, 1, srgb_nonlinear=False)

        self.viewport: Viewport = viewport
        self.shadow_viewport: Viewport = shadow_viewport
        
        self.asset_manager: AssetManager = AssetManager()
        self.framebuffer: Framebuffer = Framebuffer(
            color_attachment, resolve_attachment, depth_attachment)
        
        self.shadow_framebuffer: Framebuffer = Framebuffer(
            color_attachment=None, resolve_attachment=None, depth_attachment=shadow_map)

        self.view_matrix: Mat4 = None
        self.light_space_matrix: Mat4 = None
        self.projection_matrix: Mat4 = None

        self.models: list[list[Mesh]] = []
        self.model_transforms: list[Transform] = []

        self.point_lights: list[PointLight] = []
        self.directional_lights: list[DirectionalLight] = []
        self.spot_lights: list[SpotLight] = []

        setup_turtle(viewport.width, viewport.height)

    def add_model(self, path: str, transform: Transform):
        self.models.append(self.asset_manager.load_model(path))
        self.model_transforms.append(transform)

    def add_light(self, light):
        if (type(light) is PointLight):
            self.point_lights.append(light)
        if (type(light) is DirectionalLight):
            self.directional_lights.append(light)
        if (type(light) is SpotLight):
            self.spot_lights.append(light)
            self.light_space_matrix = self.projection_matrix * make_lookat_matrix(light.pos, light.pos + light.dir, Vec3(0, 1, 0))

    def set_camera(self, cam: Camera) -> int:
        ar: float = self.viewport.width / self.viewport.height

        self.view_matrix = make_lookat_matrix(
            cam.pos, cam.target, Vec3(0, 1, 0))
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
            
            shadow_pass_vertex_uniforms = ShadowPassVertexShader.Uniforms(
                    model_matrix=model_matrix, 
                    light_space_matrix=self.light_space_matrix
            )
            shadow_pass_vertex_shader = ShadowPassVertexShader(shadow_pass_vertex_uniforms)
            shadow_pass_program = ShaderProgram(
                    vertex_shader=shadow_pass_vertex_shader, fragment_shader=None
            )

            phong_vertex_uniforms = PhongVertexShader.Uniforms(
                model_matrix=model_matrix, normal_matrix=normal_matrix,
                view_matrix=self.view_matrix, projection_matrix=self.projection_matrix, 
                light_space_matrix=self.light_space_matrix
            )

            phong_vertex_shader = PhongVertexShader(phong_vertex_uniforms)
            for mesh in model:
                vertex_buffer = {"pos": mesh.positions,
                    "normal": mesh.normals, "tex_uv": mesh.tex_uvs}
                
                draw(self.shadow_framebuffer, self.shadow_viewport, shadow_pass_program,
                    vertex_buffer, 0, mesh.num_vertices)
                
                material: Material = mesh.material

                phong_fragment_uniforms = PhongFragmentShader.Uniforms(
                    material=material,
                    point_lights=self.point_lights, directional_lights=self.directional_lights,
                    spot_lights=self.spot_lights, 
                    shadow_map=self.shadow_framebuffer.depth_attachment
                )
                phong_fragment_shader = PhongFragmentShader(phong_fragment_uniforms)

                phong_program = ShaderProgram(phong_vertex_shader, phong_fragment_shader)

                draw(self.framebuffer, self.viewport, phong_program,
                     vertex_buffer, 0, mesh.num_vertices)

        resolve_buffer(src=self.framebuffer.color_attachment,
                       target=self.framebuffer.resolve_attachment)

    def present(self):
        present_backbuffer(self.framebuffer.resolve_attachment, self.viewport)

    def finish(self):
        turtle.done()
