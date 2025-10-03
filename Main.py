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
    texture_uv: Vec4 

backbuffer_image = [[Vec4(1.0, 1.0, 1.0, 1.0) for x in range(WINDOW_WIDTH)]
                for y in range(WINDOW_HEIGHT)]

depth_buffer_image = [[999999.0 for x in range(WINDOW_WIDTH)]
                for y in range(WINDOW_HEIGHT)]

texture_image = [[Vec4(1.0, 1.0, 1.0, 1.0) for x in range(WINDOW_WIDTH)]
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

    min_x_px = max(min_x_px, 0)
    max_x_px = min(max_x_px, WINDOW_WIDTH)
    min_y_px = max(min_y_px, 0)
    max_y_px = min(max_y_px, WINDOW_HEIGHT)

    for v_px in range(int(min_y_px), int(max_y_px)):
        for u_px in range(int(min_x_px), int(max_x_px)):
            u_sample: float = (u_px + 0.5)
            v_sample: float = (v_px + 0.5)
            uv: Vec4 = Vec4(u_sample, v_sample, 0.0, 1.0)

            v4: Vec4 = subtract(uv, p1.transform)
            v5: Vec4 = subtract(uv, p2.transform)

            w3: float = ((v1.x * v4.y) - (v1.y * v4.x)) / det
            w1: float = ((v2.x * v5.y) - (v2.y * v5.x)) / det
            w2: float = 1.0 - w1 - w3

            if (w1 >= 0.0 and w2 >= 0.0 and w3 >= 0.0):
                # note that .transform.w is actually original_z 
                px_depth: float = 1.0 / (w1 / p1.transform.w + w2 / p2.transform.w + w3 / p3.transform.w)
                if (px_depth < depth_buffer_image[v_px][u_px]):
                    t1: Vec4 = scale(p1.texture_uv, w1 / p1.transform.w)
                    t2: Vec4 = scale(p2.texture_uv, w2 / p2.transform.w)
                    t3: Vec4 = scale(p3.texture_uv, w3 / p3.transform.w)
                    tuv: Vec4 = scale(add(add(t1, t2), t3), px_depth)
                    texture_color: Vec4 = texture_image[int(tuv.y * WINDOW_HEIGHT)][int(tuv.x * WINDOW_WIDTH)]
                    
                    c1: Vec4 = scale(p1.color, w1 / p1.transform.w)
                    c2: Vec4 = scale(p2.color, w2 / p2.transform.w)
                    c3: Vec4 = scale(p3.color, w3 / p3.transform.w)
                    color: Vec4 = scale(add(add(c1, c2), c3), px_depth)

                    final_color: Vec4 = add(scale(color, color.w), scale(texture_color, 1.0 - color.w))
                    
                    backbuffer_image[v_px][u_px] = final_color
                    depth_buffer_image[v_px][u_px] = px_depth


def perspective_divide(vec: Vec4) -> Vec4:
    """ 
        w component is saved.
    """
    return Vec4(vec.x / vec.w, vec.y / vec.w, vec.z / vec.w, vec.w)

def viewport_transform(ndc: Vec4, width: int, height: int) -> Vec4: 
    x_px: float = ((ndc.x + 1.0) / 2.0) * width
    y_px: float = ((ndc.y + 1.0) / 2.0) * height
    return Vec4(x_px, y_px, ndc.z, ndc.w)

def main() -> None:
    for j in range(0, WINDOW_HEIGHT):
        for i in range(0, WINDOW_WIDTH):
            if (((i // 50) % 2 + ((j // 50) % 2)) % 2 == 0):
                color: Vec4 = Vec4(0.0, 1.0, 1.0, 1.0)
                texture_image[j][i] = color
            
    transforms: list[Vec4] = [
            Vec4(-2, -2, 10.0, 1.0), Vec4(2, -2, 10.0, 1.0), Vec4(-2, 2, 20.0, 1.0),
            Vec4(2, -2, 10, 1.0), Vec4(2, 2, 20.0, 1.0), Vec4(-2, 2, 20.0, 1.0),]
    texture_uvs: list[Vec4] = [
            Vec4(0, 0, 0, 0), Vec4(1, 0, 0, 0), Vec4(0, 1, 0, 0), 
            Vec4(1, 0, 0, 0), Vec4(1, 1, 0 ,0), Vec4(0, 1, 0, 0)]
    colors: list[Vec4] = [
            Vec4(1.0, 0.0, 0.0, 0.5), Vec4(0.0, 1.0, 0.0, 0.5), Vec4(0.0, 0.0, 1.0, 0.5),
            Vec4(0.0, 1.0, 0.0, 0.5), Vec4(0.0, 0.0, 0.0, 0.0), Vec4(0.0, 0.0, 1.0, 0.5)]

    far_plane: float = 100
    near_plane: float = 0.001
    fov: float = math.radians(90/2)
    ar: float = WINDOW_WIDTH / WINDOW_HEIGHT
    projection_matrix: Mat4 = Mat4(
            Vec4(1/(ar * math.tan(fov/2)), 0, 0, 0),
            Vec4(0, 1/math.tan(fov/2), 0, 0),
            Vec4(0, 0, far_plane/(far_plane - near_plane), -(far_plane * near_plane)/(far_plane - near_plane)),
            Vec4(0, 0, 1, 0))
                              
    for i in range(0, len(transforms)): 
        transforms[i] = multiply(projection_matrix, transforms[i])
        transforms[i] = perspective_divide(transforms[i])
        transforms[i] = viewport_transform(transforms[i], WINDOW_WIDTH, WINDOW_HEIGHT)

    p1: Vertex = Vertex(transforms[0], colors[0], texture_uvs[0])
    p2: Vertex = Vertex(transforms[1], colors[1], texture_uvs[1])
    p3: Vertex = Vertex(transforms[2], colors[2], texture_uvs[2])

    p4: Vertex = Vertex(transforms[3], colors[3], texture_uvs[3])
    p5: Vertex = Vertex(transforms[4], colors[4], texture_uvs[4])
    p6: Vertex = Vertex(transforms[5], colors[5], texture_uvs[5])

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
