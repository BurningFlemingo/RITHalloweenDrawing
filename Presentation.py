import turtle
from typing import NamedTuple

from VectorMath import *
from Buffer import *


class Viewport(NamedTuple):
    width: int
    height: int


def quantize_color(color: Vec4, level: int) -> Vec4:
    return Vec4(
        round(color.x * (level - 1)) / (level - 1), round(color.y * (level - 1)) / (level - 1), round(color.z * (level - 1)) / (level - 1), 1.0)


def setup_turtle(window_width: int, window_height: int) -> None:
    turtle.setup(width=window_width, height=window_height)
    turtle.setworldcoordinates(0, 0, window_width - 1, window_height - 1)
    turtle.bgcolor(0.5, 0.5, 0.5)

    turtle.tracer(0, 0)
    turtle.pensize(1)
    

def present_backbuffer(backbuffer: Buffer, viewport: Viewport) -> None:
    pen_color: Vec4 = Vec4(0.0, 0.0, 0.0, 1.0)
    turtle.pencolor(pen_color.x, pen_color.y, pen_color.z)
    for y_px in range(0, viewport.height):
        turtle.up()
        turtle.goto(0, y_px)
        turtle.down()
        accumulated_pixels: int = 0
        for x_px in range(0, viewport.width):
            n_samples: int = backbuffer.n_samples_per_axis ** 2
            px_index = (y_px * backbuffer.width + x_px) * n_samples

            px_color: Vec4 = backbuffer.data[px_index]
            px_color = quantize_color(px_color, 16)
            color_changed: bool = px_color != pen_color

            if (color_changed):
                turtle.forward(accumulated_pixels)
                pen_color = px_color
                turtle.pencolor(pen_color.x, pen_color.y, pen_color.z)
                accumulated_pixels = 0

            accumulated_pixels += 1

        turtle.forward(accumulated_pixels)
        if (y_px % 4 == 0):
            turtle.update()
