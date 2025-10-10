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
    viewport = Viewport(WINDOW_WIDTH, WINDOW_HEIGHT)
    scene = Scene(viewport)

    transform = Transform(pos=[0, 0, 4], rot=[60, 0, -135])
    scene.add_model("assets\\test\\test.obj", transform)

    scene.render()
    scene.present()

    scene.finish()


if __name__ == "__main__":
    main()
