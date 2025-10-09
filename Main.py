import turtle
import math
import time
from typing import NamedTuple
from typing import Any

WINDOW_WIDTH: int = 1920//2
WINDOW_HEIGHT: int = 1080//2


class Vec4(NamedTuple):
    x: Any
    y: Any
    z: Any
    w: Any

    def __add__(self, other):
        return type(self)(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        return type(self)(*[a - b for a, b in zip(self, other)])

    def __mul__(self, factor):
        return type(self)(*[a * factor for a in self])

    def __truediv__(self, factor):
        return type(self)(*[a / factor for a in self])

    def __floordiv__(self, factor):
        return type(self)(*[a // factor for a in self])

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)


class Vec3(NamedTuple):
    x: Any
    y: Any
    z: Any

    def __add__(self, other):
        return type(self)(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        return type(self)(*[a - b for a, b in zip(self, other)])

    def __mul__(self, factor):
        return type(self)(*[(a * factor) for a in self])

    def __truediv__(self, factor):
        return type(self)(*[a / factor for a in self])

    def __floordiv__(self, factor):
        return type(self)(*[a // factor for a in self])

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


class Vec2(NamedTuple):
    x: Any
    y: Any

    def __add__(self, other):
        return type(self)(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        return type(self)(*[a - b for a, b in zip(self, other)])

    def __mul__(self, factor):
        return type(self)(*[(a * factor) for a in self])

    def __truediv__(self, factor):
        return type(self)(*[a / factor for a in self])

    def __floordiv__(self, factor):
        return type(self)(*[a // factor for a in self])

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)


def normalize(vec):
    return vec / vec.magnitude()


def dot(v1, v2):
    accumulated = 0
    for a, b in zip(v1, v2):
        accumulated += a * b
    return accumulated


def cross(a: Vec3, b: Vec3):
    return Vec3(
        a.y * b.z - a.z * b.y,
        -(a.x * b.z - a.z * b.x),
        a.x * b.y - a.y * b.x
    )


def hadamard(v1, v2):
    return type(v1)(*[a * b for a, b in zip(v1, v2)])


def reflect(incoming, normal):
    return incoming - (normal * (2 * dot(normal, incoming)))


class Mat4(NamedTuple):
    row1: Vec4
    row2: Vec4
    row3: Vec4
    row4: Vec4

    def __mul__(self, other):
        if isinstance(other, Mat4):
            transposed = transpose(other)
            rows: list[Vec4] = []
            for row in self:
                rows.append(Vec4(*[dot(row, col) for col in transposed]))
            return Mat4(*rows)
        else:
            return Vec4(*[dot(row, other) for row in self])


def transpose(mat: Mat4):
    return Mat4(
        Vec4(mat.row1.x, mat.row2.x, mat.row3.x, mat.row4.x),
        Vec4(mat.row1.y, mat.row2.y, mat.row3.y, mat.row4.y),
        Vec4(mat.row1.z, mat.row2.z, mat.row3.z, mat.row4.z),
        Vec4(mat.row1.w, mat.row2.w, mat.row3.w, mat.row4.w))


class Rot3(NamedTuple):
    scalar: float
    xy: float
    yz: float
    zx: float

    def __mul__(self, other):
        return Rot3(
            self.scalar * other.scalar - self.xy * other.xy -
            self.yz * other.yz - self.zx * other.zx,
            self.scalar * other.xy + self.xy * other.scalar -
            self.yz * other.zx + self.zx*other.yz,
            self.scalar * other.yz + self.xy * other.zx +
            self.yz * other.scalar - self.zx * other.xy,
            self.scalar * other.zx - self.xy * other.yz +
            self.yz * other.xy + self.zx * other.scalar
        )


def wedge(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(
        (a.x * b.y) - (a.y * b.x),
        (a.y * b.z) - (a.z * b.y),
        (a.z * b.x) - (a.x * b.z)
    )


def make_rotor(from_vec: Vec3, to_vec: Vec3, theta: float = None):
    """
        equations credit: https://jacquesheunis.com/post/rotors/
    """
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)

    if (theta is None):
        # handle when halfway magnitude is zero
        halfway_vec: Vec3 = normalize(from_vec + to_vec)

        scalar_part: Vec3 = dot(from_vec, halfway_vec)
        wedge_part: Vec3 = normalize(wedge(halfway_vec, from_vec))

        return Rot3(scalar_part, *wedge_part)

    scalar_part: float = math.cos(theta / 2)
    wedge_part: Vec3 = normalize(
        wedge(to_vec, from_vec)) * math.sin(theta / 2.0)

    return Rot3(scalar_part, *wedge_part)


def make_rotor_pyr(pyr: Vec3) -> Rot3:
    """
        pyr -> pitch, yaw, roll -> x axis, y axis, z axis
        the components in pyz correspond to radians around that axis.
    """
    return \
        make_rotor(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), pyr.x) * \
        make_rotor(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), pyr.y) * \
        make_rotor(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), pyr.z)


def reverse(r: Rot3):
    """
        https://jacquesheunis.com/post/rotors/#eqn:geometric-product-complete
    """
    return Rot3(
        r.scalar,
        -r.xy,
        -r.yz,
        -r.zx
    )


def rotate(v: Vec3, r: Rot3) -> Vec3:
    """
        https://jacquesheunis.com/post/rotors/#eqn:geometric-product-complete
    """

    s_x: float = r.scalar * v.x + r.xy * v.y - r.zx * v.z
    s_y: float = r.scalar * v.y - r.xy * v.x + r.yz * v.z
    s_z: float = r.scalar * v.z - r.yz * v.y + r.zx * v.x
    s_xyz: float = r.xy * v.z + r.yz * v.x + r.zx * v.y

    return Vec3(
        s_x * r.scalar + s_y * r.xy + s_xyz * r.yz - s_z * r.zx,
        s_y * r.scalar - s_x * r.xy + s_z * r.yz + s_xyz * r.zx,
        s_z * r.scalar + s_xyz * r.xy - s_y * r.yz + s_x * r.zx
    )


def make_rotation_matrix(r: Rot3) -> Mat4:
    x_basis: Vec3 = rotate(Vec3(1.0, 0.0, 0.0), r)
    y_basis: Vec3 = rotate(Vec3(0.0, 1.0, 0.0), r)
    z_basis: Vec3 = rotate(Vec3(0.0, 0.0, 1.0), r)
    return Mat4(
        Vec4(x_basis.x, y_basis.x, z_basis.x, 0.0),
        Vec4(x_basis.y, y_basis.y, z_basis.y, 0.0),
        Vec4(x_basis.z, y_basis.z, z_basis.z, 0.0),
        Vec4(0.0, 0.0, 0.0, 1.0)
    )


class Buffer(NamedTuple):
    data: list[Any]
    width: int
    height: int
    n_samples_per_axis: int

    # writes to all samples
    def write2D(self, x: int, y: int, val: Vec4):
        samples: int = self.n_samples_per_axis ** 2
        for sample in range(0, samples):
            self.data[samples * (y * self.width + x) + sample] = val

    def sampleUV(self, u: float, v: float):
        """
            u and v should be normalized between 0 and 1.
        """
        n_samples: int = self.n_samples_per_axis ** 2

        x: int = int(u * (self.width - 1))
        y: int = int(v * (self.height - 1))

        index: int = (y * self.width + x) * n_samples

        return self.data[index]


class Framebuffer(NamedTuple):
    color_attachment: Buffer
    depth_attachment: Buffer


class Vertex(NamedTuple):
    transform: Vec4
    attrib: tuple


class Model(NamedTuple):
    vertices: tuple[Vec4, Vec3, Vec2]
    texture: Buffer


class Material(NamedTuple):
    ambient_strength: float
    specular_strength: float
    shininess: float


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
    if (edge.y < 0):
        return True

    if (edge.x < 0 and edge.y == 0):
        return True

    return False


def test_samples(ctx: RasterCtx, u_px: int, v_px: int, w1: int, w2: int) -> (list[int], int, int):
    fb, p1, p2, p3, det, w1_px_step, w2_px_step, w1_bias, w2_bias, w3_bias = ctx
    n_samples_per_axis: int = fb.depth_attachment.n_samples_per_axis
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
                interpolated_depth: float = (p1.transform.z *
                                             w1 + p2.transform.z * w2 + p3.transform.z * w3) / det

                sample_index: int = v_sample * n_samples_per_axis + u_sample
                depth_buffer_index: int = px_index + sample_index

                # print(v_px, u_px, depth_buffer_index)

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


def interpolate_attributes(p1_attrib: tuple, p2_attrib: tuple, p3_attrib: tuple, w1: float, w2: float, w3: float, px_depth: float) -> tuple:

    n_attributes: int = len(p1_attrib)
    attributes = []
    for attrib_index in range(0, n_attributes):
        a1 = p1_attrib[attrib_index] * w1
        a2 = p2_attrib[attrib_index] * w2
        a3 = p3_attrib[attrib_index] * w3
        interpolated = (a1 + a2 + a3) * px_depth
        attributes.append(interpolated)

    return tuple(attributes)


def fragment_shader(uniforms: tuple, attributes: tuple) -> Vec4:
    texture, material = uniforms
    normal, tex_uv, position, light_pos = attributes

    light_color: Vec3 = Vec3(1.0, 0.3, 1.0)

    norm: float = normalize(normal)

    view_dir: Vec3 = normalize(position * -1)

    light_dir: Vec3 = normalize(light_pos - position)

    ambient: Vec3 = light_color * material.ambient_strength

    diffuse_strength: float = max(dot(light_dir, norm), 0)
    diffuse: Vec3 = light_color * diffuse_strength

    halfway: Vec3 = normalize(view_dir + light_dir)
    spec: float = max(dot(norm, halfway), 0) ** material.shininess
    specular = light_color * (spec * material.specular_strength)

    object_color: Vec3 = Vec3(*texture.sampleUV(*tex_uv)[:3])
    result: Vec3 = hadamard(
        ambient + diffuse + specular + Vec3(0.3, 0.3, 0.3), object_color)

    return Vec4(*result, 1.0)


def shade_pixel(ctx: RasterCtx, uniforms: tuple, u_px: int, v_px: int, w1: int, w2: int) -> bool:
    fb, p1, p2, p3, det, w1_px_step, w2_px_step, w1_bias, w2_bias, w3_bias = ctx

    samples_survived_indices, accumulated_w1, accumulated_w2 = test_samples(
        ctx, u_px, v_px, w1, w2)

    n_surviving_samples: int = len(samples_survived_indices)
    if (n_surviving_samples == 0):
        return False

    w1 = accumulated_w1 / (n_surviving_samples * det)
    w2 = accumulated_w2 / (n_surviving_samples * det)
    w3 = 1.0 - w1 - w2
    px_depth: float = 1.0 / (w1/p1.transform.w +
                             w2/p2.transform.w + w3/p3.transform.w)

    interpolated_attributes: tuple = interpolate_attributes(
        p1.attrib, p2.attrib, p3.attrib, w1, w2, w3, px_depth)

    color: Vec4 = fragment_shader(uniforms, interpolated_attributes)
    color = Vec4(
        min(max(color.x, 0.0), 1.0),
        min(max(color.y, 0.0), 1.0),
        min(max(color.z, 0.0), 1.0),
        min(max(color.w, 0.0), 1.0))

    n_samples: int = fb.color_attachment.n_samples_per_axis ** 2
    px_index: int = (v_px * fb.color_attachment.width + u_px) * \
        n_samples
    for sample_index in samples_survived_indices:
        fb.color_attachment.data[px_index + sample_index] = color

    return True


def resolve_buffer(buffer: Buffer) -> None:
    n_samples: int = buffer.n_samples_per_axis ** 2
    for j in range(0, buffer.height):
        for i in range(0, buffer.width):
            px_index: int = (j * buffer.width + i) * n_samples
            accumulated_value = buffer.data[px_index]
            for sample_index in range(1, n_samples):
                accumulated_value = accumulated_value + \
                    buffer.data[px_index + sample_index]
            average_value = accumulated_value * (1/n_samples)
            for sample_index in range(0, n_samples):
                buffer.data[px_index + sample_index] = average_value


def subpx_transform(point: Vec4, n_sub_px_per_axis: int) -> Vec4:
    return Vec4(round(point.x * n_sub_px_per_axis), round(point.y * n_sub_px_per_axis), point.z, point.w)


def attrib_pre_divide(p: Vertex) -> tuple:
    return tuple([attrib / p.transform.w for attrib in p.attrib])


def rasterize_triangle(fb: Framebuffer, uniforms: tuple, p1: Vertex, p2: Vertex, p3: Vertex) -> bool:
    n_subpx_per_axis: int = 256

    # pre-dividing so inner loop isnt calculating this a ton for no reason
    p1 = Vertex(
        subpx_transform(p1.transform, n_subpx_per_axis),
        attrib_pre_divide(p1))
    p2 = Vertex(
        subpx_transform(p2.transform, n_subpx_per_axis),
        attrib_pre_divide(p2))
    p3 = Vertex(
        subpx_transform(p3.transform, n_subpx_per_axis),
        attrib_pre_divide(p3))

    edge1: Vec4 = p2.transform - p1.transform
    edge2: Vec4 = p3.transform - p2.transform
    edge3: Vec4 = p1.transform - p3.transform

    w1_bias: int = 1 if is_covered_edge(edge2) else 0
    w2_bias: int = 1 if is_covered_edge(edge3) else 0
    w3_bias: int = 1 if is_covered_edge(edge1) else 0

    det: int = (edge1.x * edge2.y) - (edge1.y * edge2.x)
    if (det <= 0):
        return False

    min_x_px: int = math.floor(min(
        min(p1.transform.x, p2.transform.x), p3.transform.x) / n_subpx_per_axis)
    max_x_px: int = math.ceil(max(
        max(p1.transform.x, p2.transform.x), p3.transform.x) / n_subpx_per_axis)
    min_y_px: int = math.floor(min(
        min(p1.transform.y, p2.transform.y), p3.transform.y) / n_subpx_per_axis)
    max_y_px: int = math.ceil(max(
        max(p1.transform.y, p2.transform.y), p3.transform.y) / n_subpx_per_axis)

    if (min_x_px < 0 or max_x_px > WINDOW_WIDTH or min_y_px < 0 or max_y_px > WINDOW_HEIGHT):
        return False

    w1_px_step: Vec2 = Vec2(int(-edge2.y) * n_subpx_per_axis,
                            int(edge2.x) * n_subpx_per_axis)
    w2_px_step: Vec2 = Vec2(int(-edge3.y) * n_subpx_per_axis,
                            int(edge3.x) * n_subpx_per_axis)

    initial_uv: Vec4 = Vec4(min_x_px * n_subpx_per_axis,
                            min_y_px * n_subpx_per_axis, 0, 1)

    v5: Vec4 = initial_uv - p2.transform
    v6: Vec4 = initial_uv - p3.transform

    w1: int = (int(edge2.x) * int(v5.y)) - (int(edge2.y) * int(v5.x))
    w2: int = (int(edge3.x) * int(v6.y)) - (int(edge3.y) * int(v6.x))

    ctx: RasterCtx = RasterCtx(
        fb, p1, p2, p3, det, w1_px_step, w2_px_step, w1_bias, w2_bias, w3_bias)

    for v_px in range(min_y_px, max_y_px):
        row_w1: float = w1
        row_w2: float = w2
        for u_px in range(min_x_px, max_x_px):
            shade_pixel(ctx, uniforms, u_px, v_px, w1, w2)

            w1 += w1_px_step.x
            w2 += w2_px_step.x
        w1 = row_w1 + w1_px_step.y
        w2 = row_w2 + w2_px_step.y

    return True


def perspective_divide(vec: Vec4) -> Vec4:
    """
        w component is saved.
    """
    return Vec4(vec.x / vec.w, vec.y / vec.w, vec.z / vec.w, vec.w)


def load_bmp(path: str) -> Buffer:
    with open(path, 'rb') as bmp:
        loaded_bmp: bytes = bmp.read()

        pixel_array_offset: int = int.from_bytes(
            loaded_bmp[10:14], byteorder="little")
        width: int = int.from_bytes(loaded_bmp[18:22], byteorder="little")
        height: int = int.from_bytes(loaded_bmp[22:26], byteorder="little")

        bpp: int = int.from_bytes(
            loaded_bmp[28:30], byteorder="little")  # bpp: bits per pixel
        compression_method: int = int.from_bytes(
            loaded_bmp[30:34], byteorder="little")
        n_colors_in_pallet: int = int.from_bytes(
            loaded_bmp[46:50], byteorder="little")

        if (compression_method != 0 or n_colors_in_pallet != 0 or bpp <= 16):
            print(f"{path} is formatted wrong: compression_method: {
                  compression_method}, n_colors_in_pallet: {n_colors_in_pallet}, bpp: {bpp}")

        row_size: int = ((bpp * width + 31) // 32) * 4  # padding included
        pixel_array_size: int = row_size * height

        bytes_per_pixel: int = bpp // 8
        n_color_channels: int = 3
        if (bytes_per_pixel % 4 == 0):
            n_color_channels = 4

        bytes_per_color_channel: int = bytes_per_pixel // n_color_channels

        raw_pixels: bytes = loaded_bmp[pixel_array_offset:
                                       pixel_array_offset + pixel_array_size]
        pixels: list[Vec4] = [
            Vec4(0, 0, 0, 0) for a in range(width * height)]

        channel_max: float = ((2 ** (bytes_per_color_channel * 8)) - 1)
        channel_inv_max: float = 1.0 / channel_max

        for row in range(height):
            row_offset: int = row * width
            row_byte_offset: int = row * row_size

            for column in range(0, width):
                offset: int = row_byte_offset + (column * bytes_per_pixel)
                b: int = raw_pixels[offset]
                g: int = raw_pixels[offset + 1]
                r: int = raw_pixels[offset + 2]
                if (n_color_channels == 4):
                    a: int = raw_pixels[offset + 3]
                else:
                    a: int = channel_max

                pixels[row_offset + column] = Vec4(
                    r * channel_inv_max, g * channel_inv_max, b *
                    channel_inv_max, a * channel_inv_max
                )

            if row % 100 == 0:
                print("Read row", row, "of", path)

        return Buffer(pixels, width, height, 1)


def load_obj(path: str) -> (list[Vec4], list[Vec4], list[Vec2]):
    unique_transforms: list[Vec4] = []
    unique_normals: list[Vec4] = []
    unique_tex_uvs: list[Vec2] = []

    transforms: list[Vec4] = []
    normals: list[Vec4] = []
    tex_uvs: list[Vec2] = []
    with open(path) as obj:
        for line in obj:
            items: [str] = line.strip().split()
            if (len(items) == 0):
                continue

            id: str = items[0]

            if (id == '#'):
                continue
            elif (id == 'v'):
                # -z to do right-handed to left-handed coord system change
                transform: Vec4 = Vec4(float(items[1]), float(
                    items[2]), -float(items[3]), 1.0)
                unique_transforms.append(transform)
            elif (id == "vt"):
                tex_uv: Vec2 = Vec2(float(items[1]), float(items[2]))
                unique_tex_uvs.append(tex_uv)
            elif (id == "vn"):
                # -z to do right-handed to left-handed coord system change
                normal: Vec4 = Vec4(float(items[1]), float(
                    items[2]), -float(items[3]), 1.0)
                unique_normals.append(normal)
            elif (id == "f"):
                atribs: list[list[int]] = [
                    [int(attribute_index) - 1 for attribute_index in vertex.split("/")] for vertex in items[1:]]
                transforms.append(unique_transforms[atribs[0][0]])
                transforms.append(unique_transforms[atribs[1][0]])
                transforms.append(unique_transforms[atribs[2][0]])

                normals.append(unique_normals[atribs[0][2]])
                normals.append(unique_normals[atribs[1][2]])
                normals.append(unique_normals[atribs[2][2]])

                tex_uvs.append(unique_tex_uvs[atribs[0][1]])
                tex_uvs.append(unique_tex_uvs[atribs[1][1]])
                tex_uvs.append(unique_tex_uvs[atribs[2][1]])

    return (transforms, normals, tex_uvs)


def viewport_transform(ndc: Vec4, width: int, height: int) -> Vec4:
    x_px: float = ((ndc.x + 1.0) * 0.5) * width
    y_px: float = ((ndc.y + 1.0) * 0.5) * height
    return Vec4(x_px, y_px, ndc.z, ndc.w)


def quantize_color(color: Vec4, level: int) -> Vec4:
    return Vec4(
        round(color.x * (level - 1)) / (level - 1), round(color.y * (level - 1)) / (level - 1), round(color.z * (level - 1)) / (level - 1), 1.0)


def present_backbuffer(backbuffer: Buffer) -> None:
    pen_color: Vec4 = Vec4(0.0, 0.0, 0.0, 1.0)
    turtle.pencolor(pen_color.x, pen_color.y, pen_color.z)
    start_time = time.time()
    for y_px in range(0, WINDOW_HEIGHT):
        turtle.up()
        turtle.goto(0, y_px)
        turtle.down()
        accumulated_pixels: int = 0
        for x_px in range(0, WINDOW_WIDTH):
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

    end_time = time.time()
    print("Backbuffer presentation took", end_time - start_time, "seconds")


def setup_turtle() -> None:
    turtle.setup(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    turtle.setworldcoordinates(0, 0, WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1)
    turtle.bgcolor(0.5, 0.5, 0.5)

    turtle.tracer(0, 0)
    turtle.pensize(1)


def draw(framebuffer: Framebuffer, vertices: tuple[list], uniforms: tuple):
    print("draw call started")
    texture, material, projection_matrix, model_matrix, rot_matrix, light_pos = uniforms
    transforms, normals, texture_uvs = vertices
    vertices: list[Vertex] = []
    for i in range(0, len(transforms)):
        # vertex shader
        transform: Vec4 = model_matrix * rot_matrix * transforms[i]
        normal: Vec3 = Vec3(*(rot_matrix * normals[i])[:3])

        attribs: tuple = (
            normal, texture_uvs[i], Vec3(*transform[:3]), light_pos)

        transform = projection_matrix * transform

        transform = perspective_divide(transform)
        transform = viewport_transform(
            transform, WINDOW_WIDTH, WINDOW_HEIGHT)

        vertex: Vertex = Vertex(transform, attribs)
        vertices.append(vertex)

    fragment_uniforms: tuple = (texture, material)

    for i in range(0, len(vertices), 3):
        v1: Vertex = vertices[i]
        v2: Vertex = vertices[i + 1]
        v3: Vertex = vertices[i + 2]

        rasterize_triangle(
            framebuffer, fragment_uniforms, v1, v2, v3)


def main() -> None:
    n_samples_per_axis: int = 2

    x_rot_angle: float = math.radians(60)
    y_rot_angle: float = math.radians(0)
    z_rot_angle: float = math.radians(-135)

    cube: Model = Model(load_obj("pumpkin.obj"), load_bmp("pumpkin.bmp"))
    house: Model = Model(load_obj("test.obj"), load_bmp("test.bmp"))

    color_attachment = Buffer([Vec4(0.1, 0.1, 0.1, 1.0) for x in range(WINDOW_WIDTH * WINDOW_HEIGHT * (n_samples_per_axis ** 2))],
                              WINDOW_WIDTH, WINDOW_HEIGHT, n_samples_per_axis)

    depth_attachment = Buffer([float("inf") for x in range(WINDOW_WIDTH * WINDOW_HEIGHT * (n_samples_per_axis ** 2))],
                              WINDOW_WIDTH, WINDOW_HEIGHT, n_samples_per_axis)

    far_plane: float = 100
    near_plane: float = 0.001
    fov: float = math.radians(90/2)
    ar: float = WINDOW_WIDTH / WINDOW_HEIGHT
    projection_matrix: Mat4 = Mat4(
        Vec4(1/(ar * math.tan(fov/2)), 0, 0, 0),
        Vec4(0, 1/math.tan(fov/2), 0, 0),
        Vec4(0, 0, far_plane/(far_plane - near_plane), -
             (far_plane * near_plane)/(far_plane - near_plane)),
        Vec4(0, 0, 1, 0))

    model_matrix: Mat4 = Mat4(
        Vec4(1, 0, 0, 0),
        Vec4(0, 1, 0, 0),
        Vec4(0, 0, 1, 4),
        Vec4(0, 0, 0, 1))

    light_pos: Vec3 = Vec3(-1.0, 0, 0.0)
    model_matrix_2: Mat4 = Mat4(
        Vec4(0.1, 0, 0, -0.5),
        Vec4(0, 0.1, 0, 0),
        Vec4(0, 0, 0.1, 2.2),
        Vec4(0, 0, 0, 1))

    rot_matrix: Mat4 = make_rotation_matrix(
        make_rotor_pyr(Vec3(x_rot_angle, y_rot_angle, z_rot_angle)))

    house_mat: Material = Material(0.0, 1.0, 32)
    light_mat: Material = Material(0.7, 0.0, 32)

    uniforms: tuple = (house.texture, house_mat, projection_matrix,
                       model_matrix, rot_matrix, light_pos)
    framebuffer: Framebuffer = Framebuffer(
        color_attachment, depth_attachment)

    draw(framebuffer, house.vertices, uniforms)

    # uniforms: tuple = (cube.texture, house_mat, projection_matrix,
    #                    model_matrix_2, rot_matrix, light_pos)
    # draw(framebuffer, cube.vertices, uniforms)

    setup_turtle()
    resolve_buffer(framebuffer.color_attachment)
    present_backbuffer(framebuffer.color_attachment)

    print("DONE!!!")

    turtle.done()


if __name__ == "__main__":
    main()
