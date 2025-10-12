from typing import Callable

from VectorMath import *
from MatrixMath import *
from RenderTypes import *


FragmentShader = Callable[[Any], list[Vec4]]


class RasterCtx(NamedTuple):
    fb: Framebuffer

    p1: Vertex
    p2: Vertex
    p3: Vertex

    det: int

    w1_px_step: Vec2
    w2_px_step: Vec2

    w1_bias: float
    w2_bias: float
    w3_bias: float


def is_covered_edge(edge: Vec4) -> bool:
    # top-left rule
    if (edge.y < 0):
        return True

    if (edge.x < 0 and edge.y == 0):
        return True

    return False


def test_samples(ctx: RasterCtx, u_px: int, v_px: int, w1: int, w2: int) -> (list[int], int, int):
    fb, p1, p2, p3, det, w1_px_step, w2_px_step, w1_bias, w2_bias, w3_bias = ctx
    n_samples_per_axis: int = fb.n_samples_per_axis
    n_samples: int = n_samples_per_axis ** 2

    px_index: int = (v_px * fb.depth_attachment.width +
                     u_px) * n_samples

    accumulated_w1: int = 0
    accumulated_w2: int = 0

    w1_sample_step: Vec2 = Vec2(w1_px_step.x // n_samples_per_axis,
                                w1_px_step.y // n_samples_per_axis)
    w2_sample_step: Vec2 = Vec2(w2_px_step.x // n_samples_per_axis,
                                w2_px_step.y // n_samples_per_axis)

    w1 += (w1_sample_step.x + w1_px_step.y) // (n_samples_per_axis * 2)
    w2 += (w2_sample_step.x + w2_px_step.y) // (n_samples_per_axis * 2)

    samples_survived_indices: list[int] = []
    for v_sample in range(0, n_samples_per_axis):
        row_w1: int = w1
        row_w2: int = w2
        for u_sample in range(0, n_samples_per_axis):
            w3: int = det - w1 - w2

            if (((w1 + w1_bias) | (w2 + w2_bias) | (w3 + w3_bias)) > 0):
                interpolated_depth: float = (
                    p1.pos.z * w1 + p2.pos.z * w2 + p3.pos.z * w3) / det

                sample_index: int = v_sample * n_samples_per_axis + u_sample
                depth_buffer_index: int = px_index + sample_index

                if (interpolated_depth <= fb.depth_attachment.data[depth_buffer_index]):

                    fb.depth_attachment.data[depth_buffer_index] = interpolated_depth
                    samples_survived_indices.append(sample_index)

                    accumulated_w1 += w1
                    accumulated_w2 += w2

            w1 += w1_sample_step.x
            w2 += w2_sample_step.x

        w1 = row_w1 + w1_sample_step.y
        w2 = row_w2 + w2_sample_step.y

    return (samples_survived_indices, accumulated_w1, accumulated_w2)


def interpolate_attributes(p1_attrib: NamedTuple, p2_attrib: NamedTuple, p3_attrib: NamedTuple, w1: float, w2: float, w3: float, px_depth: float) -> NamedTuple:

    n_attributes: int = len(p1_attrib)
    attributes = []
    for attrib_index in range(0, n_attributes):
        a1 = p1_attrib[attrib_index] * w1
        a2 = p2_attrib[attrib_index] * w2
        a3 = p3_attrib[attrib_index] * w3
        interpolated = (a1 + a2 + a3) * px_depth
        attributes.append(interpolated)

    return type(p1_attrib)(*attributes)


def shade_pixel(ctx: RasterCtx, fragment_shader: FragmentShader, u_px: int, v_px: int, w1: int, w2: int) -> bool:
    fb, p1, p2, p3, det, w1_px_step, w2_px_step, w1_bias, w2_bias, w3_bias = ctx

    samples_survived_indices, accumulated_w1, accumulated_w2 = test_samples(
        ctx, u_px, v_px, w1, w2)
    if (fb.color_attachments is None):
        return False

    n_surviving_samples: int = len(samples_survived_indices)
    if (n_surviving_samples == 0):
        return False

    w1 = accumulated_w1 / (n_surviving_samples * det)
    w2 = accumulated_w2 / (n_surviving_samples * det)
    w3 = 1.0 - w1 - w2
    px_depth: float = 1.0 / (w1/p1.pos.w +
                             w2/p2.pos.w + w3/p3.pos.w)

    interpolated_attributes: NamedTuple = interpolate_attributes(
        p1.fragment_attributes, p2.fragment_attributes, p3.fragment_attributes, w1, w2, w3, px_depth)

    colors: list[Vec4] = fragment_shader(interpolated_attributes)
    n_samples: int = fb.n_samples_per_axis ** 2
    for i in range(0, len(colors)):
        color: Vec4 = colors[i]
        color = Vec4(
            min(max(color.x, 0.0), 1.0),
            min(max(color.y, 0.0), 1.0),
            min(max(color.z, 0.0), 1.0),
            min(max(color.w, 0.0), 1.0))

        px_index: int = (
            v_px * fb.color_attachments[i].width + u_px) * n_samples
        for sample_index in samples_survived_indices:
            fb.color_attachments[i].data[px_index + sample_index] = color

    return True


def subpx_transform(point: Vec4, n_sub_px_per_axis: int) -> Vec4:
    return Vec4(round(point.x * n_sub_px_per_axis), round(point.y * n_sub_px_per_axis), point.z, point.w)


def attrib_pre_divide(p: Vertex) -> NamedTuple:
    if (p.fragment_attributes is None):
        return
    return type(p.fragment_attributes)(*[attrib / p.pos.w for attrib in p.fragment_attributes])


def rasterize_triangle(fb: Framebuffer, fragment_shader: FragmentShader, p1: Vertex, p2: Vertex, p3: Vertex) -> bool:
    n_subpx_per_axis: int = 256

    # pre-dividing so inner loop isnt calculating this a ton for no reason
    p1 = Vertex(
        subpx_transform(p1.pos, n_subpx_per_axis),
        attrib_pre_divide(p1))
    p2 = Vertex(
        subpx_transform(p2.pos, n_subpx_per_axis),
        attrib_pre_divide(p2))
    p3 = Vertex(
        subpx_transform(p3.pos, n_subpx_per_axis),
        attrib_pre_divide(p3))

    edge1: Vec4 = p2.pos - p1.pos
    edge2: Vec4 = p3.pos - p2.pos
    edge3: Vec4 = p1.pos - p3.pos

    w1_bias: int = 1 if is_covered_edge(edge2) else 0
    w2_bias: int = 1 if is_covered_edge(edge3) else 0
    w3_bias: int = 1 if is_covered_edge(edge1) else 0

    det: int = (edge1.x * edge2.y) - (edge1.y * edge2.x)
    if (det <= 0):
        return False

    min_x_px: int = math.floor(min(
        min(p1.pos.x, p2.pos.x), p3.pos.x) / n_subpx_per_axis)
    max_x_px: int = math.ceil(max(
        max(p1.pos.x, p2.pos.x), p3.pos.x) / n_subpx_per_axis)
    min_y_px: int = math.floor(min(
        min(p1.pos.y, p2.pos.y), p3.pos.y) / n_subpx_per_axis)
    max_y_px: int = math.ceil(max(
        max(p1.pos.y, p2.pos.y), p3.pos.y) / n_subpx_per_axis)

    # add clipping so you dont have to do this ):
    if (min_x_px < 0 or max_x_px > fb.width or min_y_px < 0 or max_y_px > fb.height):
        return False

    w1_px_step: Vec2 = Vec2(int(-edge2.y) * n_subpx_per_axis,
                            int(edge2.x) * n_subpx_per_axis)
    w2_px_step: Vec2 = Vec2(int(-edge3.y) * n_subpx_per_axis,
                            int(edge3.x) * n_subpx_per_axis)

    initial_uv: Vec4 = Vec4(min_x_px * n_subpx_per_axis,
                            min_y_px * n_subpx_per_axis, 0, 1)

    v5: Vec4 = initial_uv - p2.pos
    v6: Vec4 = initial_uv - p3.pos

    w1: int = (int(edge2.x) * int(v5.y)) - (int(edge2.y) * int(v5.x))
    w2: int = (int(edge3.x) * int(v6.y)) - (int(edge3.y) * int(v6.x))

    ctx: RasterCtx = RasterCtx(
        fb, p1, p2, p3, det, w1_px_step, w2_px_step, w1_bias, w2_bias, w3_bias)

    for v_px in range(min_y_px, max_y_px):
        row_w1: float = w1
        row_w2: float = w2
        for u_px in range(min_x_px, max_x_px):
            shade_pixel(ctx, fragment_shader, u_px, v_px, w1, w2)

            w1 += w1_px_step.x
            w2 += w2_px_step.x
        w1 = row_w1 + w1_px_step.y
        w2 = row_w2 + w2_px_step.y

    return True
