from typing import NamedTuple
from dataclasses import dataclass

from VectorMath import *
from Presentation import *
from Buffer import *
from RenderTypes import *
from AssetManager import *
from MatrixMath import *
from Renderer import *
from Cubemap import *
from RenderGraph import *


from shaders.PhongLighting import *
from shaders.ShadowPass import *
from shaders.ToneMapping import *
from shaders.Quad import *
from shaders.GaussianBlur import *
from shaders.Skybox import *


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
        self.render_graph: RenderGraph = RenderGraph()

        msaa_color_attachment_info = AttachmentInfo(
            clear_value=Vec4(0.0, 0.0, 0.0, 0.0), msaa=2)
        msaa_depth_attachment_info = AttachmentInfo(
            clear_value=float("inf"), msaa=2)
        backbuffer_attachment_info = AttachmentInfo(clear_value=Vec4(
            0.5, 0.5, 0.5, 1.0), format=Format.UNORM, color_space=ColorSpace.SRGB)
        shadow_buffer_attachment_info = AttachmentInfo(clear_value=float(
            "inf"), size_mode=SizeMode.ABSOLUTE, width=200, height=200)

        self.backbuffer: Buffer = make_buffer(backbuffer_attachment_info)
        self.render_graph.set_backbuffer("backbuffer", self.backbuffer)

        shadow_pass: RenderPass = self.render_graph.add_pass(RenderPass())
        shadow_pass.set_depth_output(
            "shadow_buffer", shadow_buffer_attachment_info)

        light_pass: RenderPass = self.render_graph.add_pass(RenderPass())
        light_pass.add_input_attachment("shadow_buffer")
        light_pass.set_depth_output(
            "scene_depth_buffer", msaa_depth_attachment_info)
        light_pass.add_color_output("backbuffer", backbuffer_attachment_info)

        self.render_graph.compile()

        shadow_viewport: Viewport = Viewport(
            width=viewport.width, height=viewport.height)

        n_samples_per_axis: float = 2

        hdr_color_attachment = Buffer(
            data=[Vec4(0.1, 0.1, 0.1, 1.0) for x in range(
                viewport.width * viewport.height * (n_samples_per_axis ** 2))],
            width=viewport.width, height=viewport.height, n_samples_per_axis=n_samples_per_axis,
            format=Format.SFLOAT, color_space=ColorSpace.LINEAR
        )

        hdr_color_attachment_2 = Buffer(
            data=[Vec4(0.0, 0.0, 0.0, 0.0) for x in range(
                viewport.width * viewport.height * (n_samples_per_axis ** 2))],
            width=viewport.width, height=viewport.height, n_samples_per_axis=n_samples_per_axis,
            format=Format.SFLOAT, color_space=ColorSpace.LINEAR
        )

        pingpong_color_attachment_1 = Buffer(
            data=[Vec4(0.0, 0.0, 0.0, 0.0) for x in range(
                viewport.width * viewport.height * (n_samples_per_axis ** 2))],
            width=viewport.width, height=viewport.height, n_samples_per_axis=n_samples_per_axis,
            format=Format.SFLOAT, color_space=ColorSpace.LINEAR
        )
        pingpong_color_attachment_2 = Buffer(
            data=[Vec4(0.0, 0.0, 0.0, 0.0) for x in range(
                viewport.width * viewport.height * (n_samples_per_axis ** 2))],
            width=viewport.width, height=viewport.height, n_samples_per_axis=n_samples_per_axis,
            format=Format.SFLOAT, color_space=ColorSpace.LINEAR
        )

        hdr_resolve_attachment_1 = Buffer(
            data=[Vec3(0.0, 0.0, 0.0)
                  for x in range(viewport.width * viewport.height)],
            width=viewport.width, height=viewport.height, n_samples_per_axis=1,
            format=Format.SFLOAT, color_space=ColorSpace.LINEAR
        )
        hdr_resolve_attachment_2 = Buffer(
            data=[Vec3(0.0, 0.0, 0.0)
                  for x in range(viewport.width * viewport.height)],
            width=viewport.width, height=viewport.height, n_samples_per_axis=1,
            format=Format.SFLOAT, color_space=ColorSpace.LINEAR
        )
        ldr_color_attachment = Buffer(
            data=[Vec3(0.0, 0.0, 0.0)
                  for x in range(viewport.width * viewport.height)],
            width=viewport.width, height=viewport.height, n_samples_per_axis=1,
            format=Format.UNORM, color_space=ColorSpace.SRGB
        )

        self.skybox = load_cubemap("assets\\skybox\\")

        scene_depth_attachment = Buffer(
            data=[float("inf") for x in range(viewport.width *
                                              viewport.height * (n_samples_per_axis ** 2))],
            width=viewport.width, height=viewport.height, n_samples_per_axis=n_samples_per_axis,
            format=Format.SFLOAT, color_space=ColorSpace.LINEAR
        )

        shadow_map = Buffer(
            data=[float("inf") for x in range(
                shadow_viewport.width * shadow_viewport.height)],
            width=shadow_viewport.width, height=shadow_viewport.height, n_samples_per_axis=1,
            format=Format.SFLOAT, color_space=ColorSpace.LINEAR
        )

        self.viewport: Viewport = viewport
        self.shadow_viewport: Viewport = shadow_viewport

        self.asset_manager: AssetManager = AssetManager()

        self.shadow_framebuffer: Framebuffer = Framebuffer(
            color_attachments=None, resolve_attachments=None, depth_attachment=shadow_map,
            width=shadow_map.width, height=shadow_map.height,
            n_samples_per_axis=shadow_map.n_samples_per_axis)

        self.light_framebuffer: Framebuffer = Framebuffer(
            [hdr_color_attachment,
                pingpong_color_attachment_1], None, scene_depth_attachment,
            hdr_color_attachment.width, hdr_color_attachment.height, hdr_color_attachment.n_samples_per_axis)

        self.skybox_framebuffer: Framebuffer = Framebuffer(
            [hdr_color_attachment], [
                hdr_resolve_attachment_1], scene_depth_attachment,
            hdr_color_attachment_2.width, hdr_color_attachment_2.height, hdr_color_attachment_2.n_samples_per_axis)

        self.tonemap_framebuffer: Framebuffer = Framebuffer(
            [ldr_color_attachment], None, None,
            ldr_color_attachment.width, ldr_color_attachment.height, ldr_color_attachment.n_samples_per_axis)

        self.pingpong_framebuffers: list[Framebuffer] = [
            Framebuffer(
                [pingpong_color_attachment_1], None, None,
                pingpong_color_attachment_1.width, pingpong_color_attachment_1.height, pingpong_color_attachment_1.n_samples_per_axis
            ),
            Framebuffer(
                [pingpong_color_attachment_2], None, None,
                pingpong_color_attachment_2.width, pingpong_color_attachment_2.height, pingpong_color_attachment_2.n_samples_per_axis
            ),
        ]

        self.shadow_pipeline: GraphicsPipeline = GraphicsPipeline(
            viewport=shadow_viewport,
            framebuffer=self.shadow_framebuffer
        )

        self.light_pipeline: GraphicsPipeline = GraphicsPipeline(
            viewport=viewport,
            framebuffer=self.light_framebuffer
        )
        self.tonemap_pipeline: GraphicsPipeline = GraphicsPipeline(
            viewport=viewport,
            framebuffer=self.tonemap_framebuffer
        )
        self.skybox_pipeline: GraphicsPipeline = GraphicsPipeline(
            viewport=viewport,
            framebuffer=self.skybox_framebuffer
        )

        self.pingpong_pipelines: list[GraphicsPipeline] = [
            GraphicsPipeline(
                viewport=viewport,
                framebuffer=self.pingpong_framebuffers[0]
            ),
            GraphicsPipeline(
                viewport=viewport,
                framebuffer=self.pingpong_framebuffers[1]
            )
        ]

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

    def shadow_pass(self):
        for (model, transform) in zip(self.models, self.model_transforms):
            model_matrix: Mat4 = make_model_matrix(transform)

            vertex_shader = ShadowPassVertexShader(
                model_matrix=model_matrix,
                light_space_matrix=self.light_space_matrix
            )

            for mesh in model:
                vertex_buffer = {"pos": mesh.positions}
                draw(
                    pipeline=self.shadow_pipeline,
                    vertex_buffer=vertex_buffer,
                    vertex_shader=vertex_shader,
                    fragment_shader=None,
                    vertex_count=mesh.num_vertices,
                    vertex_offset=0
                )

    def light_pass(self):
        for (model, transform) in zip(self.models, self.model_transforms):
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
                vertex_buffer = {"pos": mesh.positions,
                                 "normal": mesh.normals, "tex_uv": mesh.tex_uvs}

                phong_fragment_shader = PhongFragmentShader(
                    material=mesh.material,
                    point_lights=self.point_lights,
                    directional_lights=self.directional_lights,
                    spot_lights=self.spot_lights,
                    shadow_map=self.shadow_framebuffer.depth_attachment,
                    skybox=self.skybox
                )

                draw(
                    pipeline=self.light_pipeline,
                    vertex_buffer=vertex_buffer,
                    vertex_shader=phong_vertex_shader,
                    fragment_shader=phong_fragment_shader,
                    vertex_count=mesh.num_vertices,
                    vertex_offset=0
                )

    def tonemap_pass(self):
        self.post_process_pass(
            self.tonemap_pipeline,
            TonemapFragmentShader(
                self.skybox_framebuffer.resolve_attachments[0]
            )
        )

    def blur_pass(self):
        self.post_process_pass(
            self.pingpong_pipelines[1],
            GaussianFragmentShader(
                self.light_framebuffer.resolve_attachments[1], True)
        )
        self.post_process_pass(
            self.pingpong_pipelines[0],
            GaussianFragmentShader(
                self.pingpong_framebuffers[1].color_attachments[0], False)
        )

    def skybox_pass(self, skybox: Cubemap):

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

        draw(
            pipeline=self.skybox_pipeline,
            vertex_buffer=vertex_buffer,
            vertex_shader=SkyboxVertexShader(
                self.view_matrix, self.projection_matrix),
            fragment_shader=SkyboxFragmentShader(skybox),
            vertex_count=len(vertex_positions),
            vertex_offset=0
        )

    def post_process_pass(self, pipeline: GraphicsPipeline, fragment_shader: FragmentShader):
        vertex_positions: list[Vec3] = [
            Vec3(1.0, 1.0, 1.0),
            Vec3(-1.0, 1.0, 1.0),
            Vec3(-1.0, -1.0, 1.0),

            Vec3(-1.0, -1.0, 1.0),
            Vec3(1.0, -1.0, 1.0),
            Vec3(1.0, 1.0, 1.0),
        ]

        vertex_buffer = {"pos": vertex_positions}

        draw(
            pipeline=pipeline,
            vertex_buffer=vertex_buffer,
            vertex_shader=QuadVertexShader(),
            fragment_shader=fragment_shader,
            vertex_count=len(vertex_positions),
            vertex_offset=0
        )

    def render(self):
        self.shadow_pass()
        self.light_pass()
        # self.skybox_pass(self.skybox)
        resolve_buffer(
            src=self.skybox_framebuffer.color_attachments[0],
            target=self.skybox_framebuffer.resolve_attachments[0]
        )
        # self.blur_pass() # expensive
        self.tonemap_pass()

    def present(self):
        present_backbuffer(
            self.tonemap_framebuffer.color_attachments[0], self.viewport)

    def finish(self):
        turtle.done()
