from typing import Callable
from threading import Thread, Barrier, local
from queue import Queue

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from Sampling import *

g_tls = local()

g_ddx_list: list[Any] = [0, 0, 0, 0]
g_ddy_list: list[Any] = [0, 0, 0, 0]


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

class QuadParams(NamedTuple):
    max_x_px: int
    max_y_px: int
    px_list: list[Vec2]
    w1_list: list[int]
    w2_list: list[int]
    semaphore: Barrier

def ddy(val: Any) -> Any:
    return 0
    id: int = g_tls.id
    g_ddy_list[id] = val
    g_tls.semaphore.wait()

    if (id == 0 or id == 2):
        return g_ddy_list[2] - g_ddy_list[0]
    return g_ddy_list[3] - g_ddy_list[1]

def ddx(val: Any) -> Any:
    return 0
    id: int = g_tls.id
    g_ddx_list[id] = val
    g_tls.semaphore.wait()

    if (id == 0 or id == 1):
        return g_ddx_list[1] - g_ddx_list[0]
    return g_ddx_list[3] - g_ddx_list[2]
    


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


def differentiate_attributes(current: Any, next: Vec2) -> Vec2:
    n_attributes: int = len(current)
    ddx: list  = []
    ddy: list  = []
    for attrib_index in range(0, n_attributes):
        dadx = next.x[attrib_index] - current[attrib_index]
        dady = next.y[attrib_index] - current[attrib_index]
        
        ddx.append(dadx)
        ddy.append(dady)

    return Vec2(type(current)(*ddx), type(current)(*ddy))

def interpolate_attributes(p1_attrib: Any, p2_attrib: Any, p3_attrib: Any, w1: float, w2: float, w3: float, px_depth: float) -> NamedTuple:
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


def quad_worker(id: int, work_queue: Queue):
    g_tls.id = id
    while True:
        work_params = work_queue.get()
        if (isinstance(work_params, Barrier)):
            work_params.wait()
            continue
        max_x_px, max_y_px, px, w1, w2, semaphore = work_params
        g_tls.semaphore = semaphore
        
        if (px.x < max_x_px and px.y < max_y_px):
            shade_pixel(g_ctx, g_fragment_shader, px.x, px.y, w1, w2)


g_quad_work_queues = [Queue() for _ in range(4)]
g_threads = None
g_ctx = None
g_fragment_shader = None

def rasterize_triangle(fb: Framebuffer, fragment_shader: FragmentShader, p1: Vertex, p2: Vertex, p3: Vertex) -> bool:
    global g_threads
    global g_barrier
    if g_threads is None:
        g_threads = [Thread(target=quad_worker, args=[i, g_quad_work_queues[i]], daemon=True) for i in range(4)]
        for thread in g_threads:
            thread.start()
    
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

    global g_ctx
    global g_fragment_shader
    g_ctx = ctx
    g_fragment_shader = fragment_shader

    semaphore = Barrier(4)
    for y_px in range(min_y_px, max_y_px - 1, 2):
        row_w1: float = w1
        row_w2: float = w2

        for x_px in range(min_x_px, max_x_px - 1, 2):
            px_list: list[Vec2] = [
                Vec2(x_px, y_px), Vec2(x_px + 1, y_px),
                Vec2(x_px, y_px + 1), Vec2(x_px + 1, y_px + 1)
            ]

            w1_list: list[int] = [
                w1, w1 + w1_px_step.x, w1 + w1_px_step.y, w1 + w1_px_step.x + w1_px_step.y
            ]
            w2_list: list[int] = [
                w2, w2 + w2_px_step.x, w2 + w2_px_step.y, w2 + w2_px_step.x + w2_px_step.y
            ]
            
            for i in range(0, len(g_quad_work_queues)):
                quad_params = (max_x_px, max_y_px, px_list[i], w1_list[i], w2_list[i], semaphore)
                g_quad_work_queues[i].put(quad_params)
            
            w1 += w1_px_step.x * 2
            w2 += w2_px_step.x * 2
        w1 = row_w1 + w1_px_step.y * 2
        w2 = row_w2 + w2_px_step.y * 2
        
    semaphore.reset()
    fence = Barrier(5)
    for queue in g_quad_work_queues:
        queue.put(fence)
    fence.wait()
        
    return True
