from typing import NamedTuple
from dataclasses import dataclass
import random

from VectorMath import *
from Presentation import *
from Buffer import *
from RenderTypes import *
from AssetManager import *
from MatrixMath import *
from Renderer import *
from Cubemap import *
from Rendergraph import *


from shaders.DepthPrePass import DepthPrePassVertexShader
from shaders.PhongLighting import *
from shaders.ShadowPass import *
from shaders.ToneMapping import *
from shaders.Quad import *
from shaders.GaussianBlur import *
from shaders.Skybox import *
from shaders.DepthPrePass import *


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
        shadow_viewport: Viewport = Viewport(
            width=viewport.width, height=viewport.height)

        self.render_graph: RenderGraph = RenderGraph()

        backbuffer_color_attachment_info = AttachmentInfo(
            width=viewport.width, height=viewport.height, size_mode=SizeMode.ABSOLUTE,
            format=Format.RGBA_UNORM, color_space=ColorSpace.SRGB
        )
        msaa_hdr_color_attachment_info = AttachmentInfo(msaa=2)
        msaa_ldr_color_attachment_info = AttachmentInfo(msaa=2, format=Format.RGBA_UNORM)
        hdr_color_attachment_info = AttachmentInfo()
        depth_attachment_info = AttachmentInfo(format=Format.D_UNORM)
        msaa_depth_attachment_info = AttachmentInfo(
            format=Format.D_UNORM, msaa=2)


        self.backbuffer: Buffer = make_buffer(backbuffer_color_attachment_info)
        backbuffer: AttachmentHandle = self.render_graph.import_attachment(self.backbuffer)
        
        shadow_map: AttachmentHandle = self.render_graph.make_attachment(depth_attachment_info)
        scene_depth_buffer: AttachmentHandle = self.render_graph.make_attachment(msaa_depth_attachment_info)
        pre_pass_scene_depth_buffer: AttachmentHandle = self.render_graph.make_attachment(msaa_depth_attachment_info)
        msaa_hdr_color_1: AttachmentHandle = self.render_graph.make_attachment(msaa_hdr_color_attachment_info)
        msaa_hdr_color_2: AttachmentHandle = self.render_graph.make_attachment(msaa_hdr_color_attachment_info)
        msaa_ldr_color: AttachmentHandle = self.render_graph.make_attachment(msaa_ldr_color_attachment_info)
        hdr_color: AttachmentHandle = self.render_graph.make_attachment(hdr_color_attachment_info)

        depth_pre_pass = self.render_graph.make_pass(viewport, self.depth_pre_pass)
        depth_pre_pass.set_depth_attachment(pre_pass_scene_depth_buffer)
        
        shadow_pass = self.render_graph.make_pass(shadow_viewport, self.shadow_pass)
        shadow_pass.set_depth_attachment(shadow_map)

        light_pass = self.render_graph.make_pass(viewport, self.light_pass)
        light_pass.set_depth_attachment(scene_depth_buffer)
        light_pass.add_input_attachment(shadow_map)
        light_pass.add_input_attachment(pre_pass_scene_depth_buffer)
        light_pass.add_color_output(msaa_hdr_color_1)

        # skybox_pass = self.render_graph.make_pass(viewport, self.skybox_pass)
        # skybox_pass.add_color_output(msaa_hdr_color_1)
        # skybox_pass.set_depth_attachment(scene_depth_buffer)

        resolve_pass = self.render_graph.make_pass(viewport, self.resolve_pass)
        resolve_pass.add_input_attachment(msaa_hdr_color_1)
        resolve_pass.add_color_output(hdr_color)

        tonemap_pass = self.render_graph.make_pass(viewport, self.tonemap_pass)
        tonemap_pass.add_input_attachment(hdr_color)
        tonemap_pass.add_color_output(backbuffer)

        self.render_graph.compile()

        self.viewport: Viewport = viewport
        self.shadow_viewport: Viewport = shadow_viewport

        self.asset_manager: AssetManager = AssetManager()

        self.view_matrix: Mat4 = None
        self.light_space_matrix: Mat4 = None
        self.projection_matrix: Mat4 = None

        self.models: list[list[Mesh]] = []
        self.model_transforms: list[Transform] = []

        self.point_lights: list[PointLight] = []
        self.directional_lights: list[DirectionalLight] = []
        self.spot_lights: list[SpotLight] = []

        self.skybox = self.asset_manager.load_cubemap("assets\\cave\\")
        self.turtle_is_setup: bool = False

        ssao_kernel: list[Vec3] = []
        n_samples: int = 64
        for i in range(0, n_samples):
            sample = Vec3(
                    random.random() * 2 - 1,
                    random.random() * 2 - 1,
                    random.random(),
            )
            sample = normalize(sample)
            scale: float = i/n_samples
            scale = 0.1 + (scale ** 2) * (0.9)
            sample *= scale 
            
            ssao_kernel.append(sample)

        ssao_noise: Buffer = Buffer(
            data=[], width=4, height=4, format=Format.RGBA_SFLOAT
                )
        for _ in range(0, 16):
            xy_rot: Vec4 = Vec4(random.random(), random.random(), 0.0, 1.0)
            ssao_noise.data.append(xy_rot)
        

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
            light_projection_mat: Mat4 = make_perspective_matrix(
                90,
                self.shadow_viewport.width / self.shadow_viewport.height,
                0.01,
                10)
            light_view_mat: Mat4 = make_lookat_matrix(
                light.pos, light.pos + light.dir, Vec3(0, 1, 0))

            self.light_space_matrix = light_projection_mat * light_view_mat

    def set_camera(self, cam: Camera):
        ar: float = self.viewport.width / self.viewport.height

        self.view_matrix = make_lookat_matrix(
            cam.pos, cam.target, Vec3(0, 1, 0))
        self.projection_matrix = make_perspective_matrix(
            cam.fov,
            ar,
            cam.near_plane,
            cam.far_plane
        )

    def depth_pre_pass(self, ctx: RenderCtx):
         for (model, transform) in zip(self.models, self.model_transforms):
            model_matrix: Mat4 = make_model_matrix(transform)

            depth_pre_pass_vertex_shader = DepthPrePassVertexShader(
                model_matrix=model_matrix,
                view_matrix=self.view_matrix,
                projection_matrix=self.projection_matrix,
            )
            for mesh in model:
                vertex_buffer = {"pos": mesh.positions}

                ctx.draw(
                    vertex_buffer=vertex_buffer,
                    vertex_shader=depth_pre_pass_vertex_shader,
                    fragment_shader=None,
                    vertex_count=mesh.num_vertices,
                    vertex_offset=0
                )
       

    def shadow_pass(self, ctx: RenderCtx):
        for (model, transform) in zip(self.models, self.model_transforms):
            model_matrix: Mat4 = make_model_matrix(transform)

            vertex_shader = ShadowPassVertexShader(
                model_matrix=model_matrix,
                light_space_matrix=self.light_space_matrix
            )

            for mesh in model:
                vertex_buffer = {"pos": mesh.positions}
                ctx.draw(
                    vertex_buffer=vertex_buffer,
                    vertex_shader=vertex_shader,
                    fragment_shader=None,
                    vertex_count=mesh.num_vertices,
                    vertex_offset=0
                )

    def light_pass(self, ctx: RenderCtx):
        for (model, transform) in zip(self.models, self.model_transforms):
            shadow_map: Sampler2D = Sampler2D([ctx.input_attachments[0]])
            pre_pass_depth_buffer: Sampler2D = Sampler2D([ctx.input_attachments[1]])

            model_matrix: Mat4 = make_model_matrix(transform)
            normal_matrix: Mat4 = make_normal_matrix(model_matrix)

            phong_vertex_shader = PhongVertexShader(
                model_matrix=model_matrix,
                normal_matrix=normal_matrix,
                view_matrix=self.view_matrix,
                projection_matrix=self.projection_matrix,
                light_space_matrix=self.light_space_matrix
            )
            for mesh in model:
                vertex_buffer = {
                    "pos": mesh.positions, "tex_uv": mesh.tex_uvs,
                    "normal": mesh.normals, "tangent": mesh.tangents, 
                    "bitangent": mesh.bitangents
                }

                phong_fragment_shader = PhongFragmentShader(
                    material=mesh.material,
                    point_lights=self.point_lights,
                    directional_lights=self.directional_lights,
                    spot_lights=self.spot_lights,
                    pre_pass_depth_buffer=pre_pass_depth_buffer,
                    shadow_map=shadow_map,
                    skybox=self.skybox
                )

                ctx.draw(
                    vertex_buffer=vertex_buffer,
                    vertex_shader=phong_vertex_shader,
                    fragment_shader=phong_fragment_shader,
                    vertex_count=mesh.num_vertices,
                    vertex_offset=0
                )

    def tonemap_pass(self, ctx: RenderCtx):
        hdr_attachment: Sampler2D = Sampler2D([ctx.input_attachments[0]])

        self.post_process_pass(
            ctx,
            TonemapFragmentShader(hdr_attachment)
        )

    def resolve_pass(self, ctx: RenderCtx):
        src_buffer: Buffer = ctx.input_attachments[0]
        target_buffer: Buffer = ctx.framebuffer.color_attachments[0]
        resolve_buffer(src_buffer, target_buffer)

    # def blur_pass(self):
    #     self.post_process_pass(
    #         self.pingpong_pipelines[1],
    #         GaussianFragmentShader(
    #             self.light_framebuffer.resolve_attachments[1], True)
    #     )
    #     self.post_process_pass(
    #         self.pingpong_pipelines[0],
    #         GaussianFragmentShader(
    #             self.pingpong_framebuffers[1].color_attachments[0], False)
    #     )

    def skybox_pass(self, ctx: RenderCtx):

        vertex_positions: list[Vec3] = [
            Vec3(-1.0,  1.0, -1.0),
            Vec3(1.0,  1.0, -1.0),
            Vec3(1.0, -1.0, -1.0),
            Vec3(1.0, -1.0, -1.0),
            Vec3(-1.0, -1.0, -1.0),
            Vec3(-1.0,  1.0, -1.0),
            Vec3(-1.0, -1.0,  1.0),
            Vec3(-1.0,  1.0,  1.0),
            Vec3(-1.0,  1.0, -1.0),
            Vec3(-1.0,  1.0, -1.0),
            Vec3(-1.0, -1.0, -1.0),
            Vec3(-1.0, -1.0,  1.0),
            Vec3(1.0, -1.0, -1.0),
            Vec3(1.0,  1.0, -1.0),
            Vec3(1.0,  1.0,  1.0),
            Vec3(1.0,  1.0,  1.0),
            Vec3(1.0, -1.0,  1.0),
            Vec3(1.0, -1.0, -1.0),
            Vec3(-1.0, -1.0,  1.0),
            Vec3(1.0,  1.0,  1.0),
            Vec3(-1.0,  1.0,  1.0),
            Vec3(1.0,  1.0,  1.0),
            Vec3(-1.0, -1.0,  1.0),
            Vec3(1.0, -1.0,  1.0),
            Vec3(-1.0,  1.0, -1.0),
            Vec3(-1.0,  1.0,  1.0),
            Vec3(1.0,  1.0,  1.0),
            Vec3(1.0,  1.0,  1.0),
            Vec3(1.0,  1.0, -1.0),
            Vec3(-1.0,  1.0, -1.0),
            Vec3(-1.0, -1.0, -1.0),
            Vec3(1.0, -1.0, -1.0),
            Vec3(-1.0, -1.0,  1.0),
            Vec3(1.0, -1.0, -1.0),
            Vec3(1.0, -1.0,  1.0),
            Vec3(-1.0, -1.0,  1.0)]

        vertex_buffer = {"pos": vertex_positions}

        ctx.draw(
            vertex_buffer=vertex_buffer,
            vertex_shader=SkyboxVertexShader(
                self.view_matrix, self.projection_matrix),
            fragment_shader=SkyboxFragmentShader(self.skybox),
            vertex_count=len(vertex_positions),
            vertex_offset=0
        )

    def post_process_pass(self, ctx: RenderCtx, fragment_shader: FragmentShader):
        vertex_positions: list[Vec3] = [
            Vec3(1.0, 1.0, 1.0),
            Vec3(-1.0, 1.0, 1.0),
            Vec3(-1.0, -1.0, 1.0),

            Vec3(-1.0, -1.0, 1.0),
            Vec3(1.0, -1.0, 1.0),
            Vec3(1.0, 1.0, 1.0),
        ]

        vertex_buffer = {"pos": vertex_positions}

        ctx.draw(
            vertex_buffer=vertex_buffer,
            vertex_shader=QuadVertexShader(),
            fragment_shader=fragment_shader,
            vertex_count=len(vertex_positions),
            vertex_offset=0
        )

    def render(self):
        self.render_graph.execute()

    def present(self):
        if (self.turtle_is_setup == False):
            self.turtle_is_setup = True
            setup_turtle(self.viewport.width, self.viewport.height)
            
        present_backbuffer(
            self.backbuffer, self.viewport)

    def finish(self):
        turtle.done()
