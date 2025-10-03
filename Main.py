import turtle
import math
from typing import NamedTuple


WINDOW_WIDTH: int = 1920//2
WINDOW_HEIGHT: int = 1080//2


class Vec4(NamedTuple):
    x: float
    y: float
    z: float
    w: float

class Mat4(NamedTuple):
    row1: Vec4
    row2: Vec4
    row3: Vec4
    row4: Vec4


class Vertex(NamedTuple):
    transform: Vec4
    color: Vec4

backbuffer_image = [[Vec4(1.0, 1.0, 1.0, 1.0) for x in range(WINDOW_WIDTH)]
                for y in range(WINDOW_HEIGHT)]

depth_buffer_image = [[999999.0 for x in range(WINDOW_WIDTH)]
                for y in range(WINDOW_HEIGHT)]

def subtract(v1: Vec4, v2: Vec4) -> Vec4:
    return Vec4(*[a - b for a, b in zip(v1, v2)])


def add(v1: Vec4, v2: Vec4) -> Vec4:
    return Vec4(*[a + b for a, b in zip(v1, v2)])

def scale(v1: Vec4, factor: float) -> Vec4:
    return Vec4(*[e * factor for e in v1])

def dot(v1: Vec4, v2: Vec4) -> float:
    accumulated: float = 0
    for a, b in zip(v1, v2):
        accumulated += a * b
    return accumulated

def multiply(mat: Mat4, vec: Vec4) -> Vec4:
    return Vec4(*[dot(row, vec) for row in mat])

def rasterize_triangle(p1: Vertex, p2: Vertex, p3: Vertex) -> None:
    v1: Vec4 = subtract(p2.transform, p1.transform)
    v2: Vec4 = subtract(p3.transform, p2.transform)
    v3: Vec4 = subtract(p1.transform, p3.transform)

    det: float = (v1.x * v2.y) - (v1.y * v2.x)
    if (abs(det) < 0.0000001): #completely magic number
        return

    min_x_px:float = min(min(p1.transform.x, p2.transform.x), p3.transform.x)
    max_x_px:float = max(max(p1.transform.x, p2.transform.x), p3.transform.x)
    min_y_px:float = min(min(p1.transform.y, p2.transform.y), p3.transform.y)
    max_y_px:float = max(max(p1.transform.y, p2.transform.y), p3.transform.y)

    for v in range(int(min_y_px), int(max_y_px)):
        for u in range(int(min_x_px), int(max_x_px)):
            u_fixed: float = (u + 0.5)
            v_fixed: float = (v + 0.5)
            uv: Vec4 = Vec4(u_fixed, v_fixed, 0.0, 1.0)

            v4: Vec4 = subtract(uv, p1.transform)
            v5: Vec4 = subtract(uv, p2.transform)

            w3: float = ((v1.x * v4.y) - (v1.y * v4.x)) / det
            w1: float = ((v2.x * v5.y) - (v2.y * v5.x)) / det
            w2: float = 1.0 - w1 - w3

            if (w1 >= 0.0 and w2 >= 0.0 and w3 >= 0.0):
                px_depth: float = 1.0 / (w1 * (1.0 / p1.transform.z) + w2 * (1.0 / p2.transform.z) + w3 * (1.0 / p3.transform.z))
                if (px_depth < depth_buffer_image[v][u]):
                    c1: Vec4 = scale(p1.color, w1)
                    c2: Vec4 = scale(p2.color, w2)
                    c3: Vec4 = scale(p3.color, w3)
                    color: Vec4 = add(add(c1, c2), c3)
                    backbuffer_image[v][u] = color
                    depth_buffer_image[v][u] = px_depth


def ndc_transform(vec: Vec4) -> Vec4:
    biased_w: float = vec.w + 1.0
    return Vec4(vec.x / biased_w, vec.y / biased_w, biased_w, 1.0)

def viewport_transform(ndc: Vec4, width: int, height: int) -> Vec4: 
    x_px: float = ((ndc.x + 1.0) / 2.0) * width
    y_px: float = ((ndc.y + 1.0) / 2.0) * height
    return Vec4(x_px, y_px, ndc.z, 1.0)

def main() -> None:
    transforms: list[Vec4] = [Vec4(-0.5, -0.5, 0, 1.0), Vec4(0.5, -0.5, 0.1, 1.0), Vec4(-0.5, 0.5, 0, 1.0), Vec4(0.5, -0.5, 0, 1.0), Vec4(0.5, 0.5, 0, 1.0), Vec4(-0.5, 0.5, 0, 1.0)]
    colors: list[Vec4] = [Vec4(1.0, 0, 0, 1.0), Vec4(0.0, 1.0, 0, 1.0), Vec4(0.0, 0, 1.0, 1.0), Vec4(1.0, 0, 0, 1.0), Vec4(0.0, 1.0, 0, 1.0), Vec4(0.0, 0, 1.0, 1.0)]

    ar: float = WINDOW_WIDTH / WINDOW_HEIGHT
    projection_matrix: Mat4 = Mat4(Vec4(1/ar, 0, 0, 0), Vec4(0, 1, 0, 0), Vec4(0, 0, 1, 0), Vec4(0, 0, 1, 0))
                              
    for i in range(0, len(transforms)): 
        transforms[i] = multiply(projection_matrix, transforms[i])
        transforms[i] = ndc_transform(transforms[i])
        transforms[i] = viewport_transform(transforms[i], WINDOW_WIDTH, WINDOW_HEIGHT)

    p1: Vertex = Vertex(transforms[0], colors[0])
    p2: Vertex = Vertex(transforms[1], colors[1])
    p3: Vertex = Vertex(transforms[2], colors[2])
                              
    p4: Vertex = Vertex(transforms[3], colors[3])
    p5: Vertex = Vertex(transforms[4], colors[4])
    p6: Vertex = Vertex(transforms[5], colors[5])

    rasterize_triangle(p1, p2, p3)
    rasterize_triangle(p4, p5, p6)
    
    turtle.setup(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    turtle.setworldcoordinates(0, 0, WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1)

    turtle.tracer(0, 0)
    turtle.pensize(1)
    
    for y_px in range(0, WINDOW_HEIGHT):
        turtle.up()
        turtle.goto(0, y_px)
        turtle.down()
        for x_px in range(0, WINDOW_WIDTH):
            color: Vec4 = backbuffer_image[y_px][x_px]
            if (color == Vec4(1.0, 1.0, 1.0, 1.0)):
                turtle.up()
            else:
                turtle.down()
                turtle.pencolor(color.x, color.y, color.z)
            turtle.forward(1)

        turtle.update()

    print("DONE!!!")

    turtle.done()


if __name__ == "__main__":
    main()
