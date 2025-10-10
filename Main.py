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
from Scene import *

WINDOW_WIDTH: int = 1920//2
WINDOW_HEIGHT: int = 1080//2


def main() -> None:
    viewport: Viewport = Viewport(WINDOW_WIDTH, WINDOW_HEIGHT)
    scene: Scene = Scene(viewport)

    x_rot_angle: float = math.radians(60)
    y_rot_angle: float = math.radians(0)
    z_rot_angle: float = math.radians(-135)

    transform: Transform = Transform(pos=Vec3(0, 0, 4), rot=make_euler_rotor(
        Vec3(x_rot_angle, y_rot_angle, z_rot_angle)), scale=Vec3(1, 1, 1))
    scene.add_model("assets\\test\\test.obj", transform)

    scene.render()
    scene.present()

    print("DONE!!!")

    turtle.done()


if __name__ == "__main__":
    main()
