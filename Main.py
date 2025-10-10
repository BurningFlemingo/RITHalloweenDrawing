import turtle
import math
import time
from typing import NamedTuple
from typing import Any

from RenderTypes import *
from Buffer import *
from VectorMath import *
from MatrixMath import *
from Rasterizer import *
from AssetLoader import *
from AssetManager import *
from Renderer import *
from Presentation import *

WINDOW_WIDTH: int = 1920//2
WINDOW_HEIGHT: int = 1080//2


def main() -> None:
    asset_manager: AssetManager = AssetManager()
    viewport: Viewport = Viewport(WINDOW_WIDTH, WINDOW_HEIGHT)

    n_samples_per_axis: int = 2

    x_rot_angle: float = math.radians(60)
    y_rot_angle: float = math.radians(0)
    z_rot_angle: float = math.radians(-135)

    color_attachment = Buffer([Vec4(0.1, 0.1, 0.1, 1.0) for x in range(WINDOW_WIDTH * WINDOW_HEIGHT * (n_samples_per_axis ** 2))],
                              WINDOW_WIDTH, WINDOW_HEIGHT, n_samples_per_axis)

    depth_attachment = Buffer([float("inf") for x in range(WINDOW_WIDTH * WINDOW_HEIGHT * (n_samples_per_axis ** 2))],
                              WINDOW_WIDTH, WINDOW_HEIGHT, n_samples_per_axis)


    framebuffer: Framebuffer = Framebuffer(color_attachment, depth_attachment)

    transform: Transform = Transform(pos=Vec3(0, 0, 4), rot=make_euler_rotor(
        Vec3(x_rot_angle, y_rot_angle, z_rot_angle)), scale=Vec3(1, 1, 1))

    model_matrix: Mat4 = make_model_matrix(transform)
    normal_matrix: Mat4 = make_normal_matrix(model_matrix)

    view_matrix: Mat4 = make_lookat_matrix(
        Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), Vec3(0, 1, 0))

    projection_matrix: Mat4 = make_projection_matrix(
        math.radians(90/2), WINDOW_WIDTH / WINDOW_HEIGHT, 0.001, 100)

    house_model: list[Mesh] = asset_manager.load_model("assets\\test\\test.obj")
    for mesh in house_model:
        material: Material = mesh.material
        magenta_light: PointLight = PointLight(pos=Vec3(
            0.0, 0.25, 4.0), linear_att=0.22, quadratic_att=0.20,  color=Vec3(1.0, 0.7, 1.0), specular=Vec3(1.0, 1.0, 1.0))

        uniforms: Uniforms = Uniforms(
            vertex=VertexUniforms(
                model_matrix=model_matrix, normal_matrix=normal_matrix,
                view_matrix=view_matrix, projection_matrix=projection_matrix),
            fragment=FragmentUniforms(
                material=material, light=magenta_light))

        draw(framebuffer, viewport, uniforms,
             (mesh.positions, mesh.normals, mesh.tex_uvs), 0, mesh.num_vertices)

    setup_turtle(*viewport)
    resolve_buffer(framebuffer.color_attachment)
    present_backbuffer(framebuffer.color_attachment, viewport)

    print("DONE!!!")

    turtle.done()


if __name__ == "__main__":
    main()
