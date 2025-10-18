from VectorMath import *
from MatrixMath import *
from RenderTypes import *


def clip_triangle(v1: Vertex, v2: Vertex, v3: Vertex) -> list[Triangle]:
    planes: list[Vec4] = [
        Vec4(0, 0, 1, 0),  # near
        Vec4(0, 0, -1, 1),  # far
        Vec4(1, 0, 0, 1),  # left
        Vec4(-1, 0, 0, 1),  # right
        Vec4(0, -1, 0, 1),  # top
        Vec4(0, 1, 0, 1)  # bottom
    ]

    triangles: list[Triangle] = [Triangle(v1, v2, v3)]

    for plane in planes:
        new_triangles: list[Triangle] = []
        for triangle in triangles:
            new_triangles += clip_triangle_against_plane(
                plane, 0, triangle)
        triangles = new_triangles

    ndc_triangles: list[Triangle] = []
    for triangle in triangles:
        v1 = perspective_divide(triangle.v1)
        v2 = perspective_divide(triangle.v2)
        v3 = perspective_divide(triangle.v3)
        ndc_triangles.append(Triangle(v1, v2, v3))

    return ndc_triangles


def clip_triangle_against_plane(plane_normal: Vec4, plane_distance: float, triangle: Triangle) -> list[Triangle]:
    vertices: list[Vertex] = []
    for i in range(0, 3):
        next_i: int = (i + 1) % 3
        current_distance: float = dot(
            plane_normal, triangle[i].pos) + plane_distance

        next_distance: float = dot(
            plane_normal, triangle[next_i].pos) + plane_distance

        current_inside: bool = current_distance >= 0
        next_inside: bool = next_distance >= 0

        if (current_inside):
            vertices.append(triangle[i])

        if (current_inside != next_inside):
            t: float = -current_distance / (next_distance - current_distance)
            new_vertex: Vertex = interpolate_vertex(
                triangle[i], triangle[next_i], t)
            vertices.append(new_vertex)

    if (len(vertices) == 3):
        return [Triangle(*vertices)]
    if (len(vertices) == 4):
        return [Triangle(*vertices[:3]), Triangle(vertices[2], vertices[3], vertices[0])]

    return []


def interpolate_vertex(va: Vertex, vb: Vertex, t: float) -> Vertex:
    pos: Vec4 = va.pos + ((vb.pos - va.pos) * t)
    if (va.fragment_attributes == None or vb.fragment_attributes == None):
        return Vertex(pos=pos, fragment_attributes=None)

    n_attributes: int = len(va.fragment_attributes)
    attributes: list | type(va.fragment_attributes) = []
    for i in range(0, n_attributes):
        interpolated = va.fragment_attributes[i] + (
            (vb.fragment_attributes[i] - va.fragment_attributes[i]) * t)
        attributes.append(interpolated)

    attributes = type(va.fragment_attributes)(*attributes)

    return Vertex(pos=pos, fragment_attributes=attributes)


def perspective_divide(v: Vertex) -> Vertex:
    """
        w component (camera space depth) is saved.
    """

    fragment_attributes: type(v.fragment_attributes) | None = None
    if (v.fragment_attributes is not None):
        fragment_attributes = \
            type(v.fragment_attributes)(
                *[attrib / v.pos.w for attrib in v.fragment_attributes])

    pos: Vec4 = Vec4(v.pos.x / v.pos.w, v.pos.y /
                     v.pos.w, v.pos.z / v.pos.w, v.pos.w)
    return Vertex(pos=pos, fragment_attributes=fragment_attributes)
