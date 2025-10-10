import turtle
from typing import NamedTuple

from VectorMath import *
from Buffer import *


class Viewport(NamedTuple):
    width: int
    height: int


def quantize_color(color: Vec3, level: int) -> Vec3:
    return Vec3(
        round(color.x * (level - 1)) / (level - 1), round(color.y * (level - 1)) / (level - 1), round(color.z * (level - 1)) / (level - 1))


def setup_turtle(window_width: int, window_height: int) -> None:
    turtle.setup(width=window_width, height=window_height)
    turtle.setworldcoordinates(0, 0, window_width - 1, window_height - 1)
    turtle.bgcolor(0.5, 0.5, 0.5)

    turtle.tracer(0, 0)
    turtle.pensize(1)


def present_backbuffer(backbuffer: Buffer, viewport: Viewport) -> None:
    pen_color: Vec3 = Vec3(0.0, 0.0, 0.0)
    turtle.pencolor(pen_color.x, pen_color.y, pen_color.z)
    for y_px in range(0, viewport.height):
        turtle.up()
        turtle.goto(0, y_px)
        turtle.down()
        accumulated_pixels: int = 0
        for x_px in range(0, viewport.width):
            px_index = y_px * backbuffer.width + x_px

            px_color: Vec3 = backbuffer.data[px_index]
            px_color = quantize_color(px_color, 32)
            color_changed: bool = px_color != pen_color

            if (color_changed):
                turtle.forward(accumulated_pixels)
                pen_color = px_color
                turtle.pencolor(*pen_color)
                accumulated_pixels = 0

            accumulated_pixels += 1

        turtle.forward(accumulated_pixels)
        if (y_px % 4 == 0):
            turtle.update()
