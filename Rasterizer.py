from typing import Callable

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from Sampling import *


FragmentShader = Callable[[Any], list[Vec4]]
        
class RasterCtx(NamedTuple):
    fb: Framebuffer

    p1: Vertex
    p2: Vertex
    p3: Vertex

    det: int

    w1_px_step: Vec2
    w2_px_step: Vec2

    w1_bias: int
    w2_bias: int
    w3_bias: int


def is_covered_edge(edge: Vec4) -> bool:
    # top-left rule
    if (edge.y < 0):
        return True

    if (edge.x < 0 and edge.y == 0):
        return True

    return False


def test_samples(ctx: RasterCtx, u_px: int, v_px: int, w1: int, w2: int) -> tuple[list[int], int, int]:
    fb, p1, p2, p3, det, w1_px_step, w2_px_step, w1_bias, w2_bias, w3_bias = ctx
    n_samples_per_axis: int = fb.n_samples_per_axis
    n_samples: int = n_samples_per_axis ** 2

    assert fb.depth_attachment != None
    
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


def interpolate_attribute(p1_attrib: Any, p2_attrib: Any, p3_attrib: Any, w1: float, w2: float, w3: float, px_depth: float) -> Any:
    a1 = p1_attrib * w1
    a2 = p2_attrib * w2
    a3 = p3_attrib * w3
    
    interpolated = (a1 + a2 + a3) * px_depth

    return interpolated 
def interpolate_attributes(p1_attrib: Any, p2_attrib: Any, p3_attrib: Any, w1: float, w2: float, w3: float, px_depth: float) -> NamedTuple:
    n_attributes: int = len(p1_attrib)
    attributes = []
    for attrib_index in range(0, n_attributes):
        interpolated: Any = interpolate_attribute(
                p1_attrib[attrib_index], p2_attrib[attrib_index], p3_attrib[attrib_index],
                w1, w2, w3, px_depth
        )
        
        attributes.append(interpolated)

    return type(p1_attrib)(*attributes)

def get_attribute(p1_attributes: Any, p2_attributes: Any, p3_attributes: Any, attribute_name: str) -> Vec3 | None:
    for i in range(0, len(p1_attributes._fields)):
        if (p1_attributes._fields[i] == attribute_name):
            return Vec3(p1_attributes[i], p2_attributes[i], p3_attributes[i])
        
    return None


def patch_attribute(attrib: Any, dudx: float, dudy: float, dvdx: float, dvdy: float) -> Any:
    if (isinstance(attrib, Sampler2D)):
        attrib.duvdx = Vec2(dudx, dvdx)
        attrib.duvdy = Vec2(dudy, dvdy)
    elif(hasattr(attrib, "_fields")):
        for i in range(0, len(attrib._fields)):
            patch_attribute(attrib[i], dudx, dudy, dvdx, dvdy)
    elif(hasattr(attrib, "__dict__") and not isinstance(attrib, type)):
        patch_samplers(attrib, dudx, dudy, dvdx, dvdy)

    return attrib

def patch_samplers(object: Any, dudx: float, dudy: float, dvdx: float, dvdy: float):
    assert object != None
    
    for attribute in object.__dict__.values():
        patch_attribute(attribute, dudx, dudy, dvdx, dvdy)


def shade_pixel(ctx: RasterCtx, fragment_shader: FragmentShader, u_px: int, v_px: int, w1: int, w2: int, ddxy: Vec2) -> bool:
    fb, p1, p2, p3, det, w1_px_step, w2_px_step, w1_bias, w2_bias, w3_bias = ctx

    n_samples: int = fb.n_samples_per_axis ** 2
    n_w1: float = 0
    n_w2: float = 0
    if (fb.depth_attachment is None):
        w1 += (w1_px_step.x + w1_px_step.y) / 2
        w2 += (w2_px_step.x + w2_px_step.y) / 2
        
        n_w1 = w1 / det
        n_w2 = w2 / det
        
        samples_survived_indices: list[int] = [i for i in range(0, n_samples)]
    else:
        samples_survived_indices, accumulated_w1, accumulated_w2 = test_samples(
            ctx, u_px, v_px, w1, w2)
        
        n_surviving_samples: int = len(samples_survived_indices)
        if (n_surviving_samples == 0):
            return False

        n_w1 = accumulated_w1 / (n_surviving_samples * det)
        n_w2 = accumulated_w2 / (n_surviving_samples * det)
        
    n_w3 = 1.0 - n_w1 - n_w2
    
    if (len(fb.color_attachments) == 0):
        return False
    
    px_depth: float = 1.0 / (n_w1/p1.pos.w +
                             n_w2/p2.pos.w + n_w3/p3.pos.w)

    interpolated_attributes: NamedTuple = interpolate_attributes(
        p1.fragment_attributes, p2.fragment_attributes, p3.fragment_attributes, n_w1, n_w2, n_w3, px_depth)

    colors: list[Vec4] = fragment_shader(interpolated_attributes)
    for i in range(0, len(colors)):
        color: Vec4 = colors[i]
        fb.color_attachments[i].write_samples(
            u_px, v_px, color, samples_survived_indices)

    return True


def subpx_transform(point: Vec4, n_sub_px_per_axis: int) -> Vec4:
    return Vec4(round(point.x * n_sub_px_per_axis), round(point.y * n_sub_px_per_axis), point.z, point.w)


def calc_duvdxy(ctx: RasterCtx, attribs: Vec3, w1: int, w2: int) -> tuple[float, float, float, float]:
    fb, p1, p2, p3, det, w1_px_step, w2_px_step, w1_bias, w2_bias, w3_bias = ctx
    
    centered_w1: float = (w1 + w1_px_step.x / 2 + w1_px_step.y / 2) / det
    centered_w2: float = (w2 + w2_px_step.x / 2 + w2_px_step.y / 2) / det
    centered_w3: float = (1 - centered_w2 - centered_w1)
    
    next_w1: Vec2 = (w1_px_step/det) + centered_w1
    next_w2: Vec2 = (w2_px_step/det) + centered_w2
    next_w3: Vec2 = Vec2(1 - next_w2.x - next_w1.x, 1 - next_w2.y - next_w1.y)
    
    current_depth: float = 1/(centered_w1/p1.pos.w + centered_w2/p2.pos.w + centered_w3/p3.pos.w)
    next_depth: Vec2 = Vec2(1/(next_w1.x/p1.pos.w + next_w2.x/p2.pos.w + next_w3.x/p3.pos.w),
        1/(next_w1.y/p1.pos.w + next_w2.y/p2.pos.w + next_w3.y/p3.pos.w))
    
    
    current_attribs: Vec2 = interpolate_attribute(
            attribs.x, attribs.y, attribs.z, centered_w1, centered_w2, centered_w3, current_depth)
    
    next_x_attribs: Vec2 = interpolate_attribute(
            attribs.x, attribs.y, attribs.z, 
            next_w1.x, next_w2.x, next_w3.x, next_depth.x
        )
        
    next_y_attribs: Vec2 = interpolate_attribute(
            attribs.x, attribs.y, attribs.z, 
            next_w1.y, next_w2.y, next_w3.y, next_depth.y
        )
    
    dudx: float = next_x_attribs.x - current_attribs.x
    dudy: float = next_y_attribs.x - current_attribs.x
    dvdx: float = next_x_attribs.y - current_attribs.y
    dvdy: float = next_y_attribs.y - current_attribs.y

    return (dudx, dudy, dvdx, dvdy)

g_ddxy_is_enabled: bool = False

def rasterize_triangle(fb: Framebuffer, fragment_shader: FragmentShader, p1: Vertex, p2: Vertex, p3: Vertex) -> bool:
    n_subpx_per_axis: int = 256

    # attributes are pre-divided in perspective divide
    p1 = Vertex(
        subpx_transform(p1.pos, n_subpx_per_axis),
        p1.fragment_attributes)
    p2 = Vertex(
        subpx_transform(p2.pos, n_subpx_per_axis),
        p2.fragment_attributes)
    p3 = Vertex(
        subpx_transform(p3.pos, n_subpx_per_axis),
        p3.fragment_attributes)

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

    ddxy = Vec2()
    for v_px in range(min_y_px, max_y_px - 1, 2):
        row_w1: float = w1
        row_w2: float = w2

        for u_px in range(min_x_px, max_x_px - 1, 2):
            if (p1.fragment_attributes != None):
                tex_uv_attribs: Vec3 | None = get_attribute(p1.fragment_attributes, p2.fragment_attributes, p3.fragment_attributes, "tex_uv")
                if (tex_uv_attribs is not None and g_ddxy_is_enabled):
                    dudx, dudy, dvdx, dvdy = calc_duvdxy(ctx, tex_uv_attribs, w1, w2)
                    patch_samplers(fragment_shader, dudx, dudy, dvdx, dvdy)

            shade_pixel(ctx, fragment_shader, u_px, v_px, w1, w2, ddxy)
            
            pixels_shaded: int = 1
            if (u_px + 1 < max_x_px):
                shade_pixel(ctx, fragment_shader, u_px + 1, v_px, w1 + w1_px_step.x, w2 + w2_px_step.x, ddxy)
                pixels_shaded += 1
            if (v_px + 1 < max_y_px):
                shade_pixel(ctx, fragment_shader, u_px, v_px + 1, w1 + w1_px_step.y, w2 + w2_px_step.y, ddxy)
                pixels_shaded += 1
            if (pixels_shaded == 3):
                shade_pixel(ctx, fragment_shader, u_px + 1, v_px + 1, w1 + w1_px_step.x + w1_px_step.y, w2 + w2_px_step.x + w2_px_step.y, ddxy)

            w1 += w1_px_step.x * 2
            w2 += w2_px_step.x * 2
        w1 = row_w1 + w1_px_step.y * 2
        w2 = row_w2 + w2_px_step.y * 2

    return True
