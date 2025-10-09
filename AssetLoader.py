from Buffer import *
from VectorMath import *


class Material_Asset(NamedTuple):
    name: str
    path: str

    ambient_color: Vec3
    diffuse_color: Vec3
    specular_color: Vec3

    ambient_map_path: str
    diffuse_map_path: str
    specular_map_path: str

class Model_Asset(NamedTuple):
    path: str
    texture_path: str

class Model(NamedTuple):
    vertices: tuple[Vec3, Vec3, Vec2]
    texture: Buffer


class Material(NamedTuple):
    diffuse: Buffer
    specular: Buffer
    shininess: float


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


def load_mtl(path: str):
    pass


def load_obj(path: str) -> (list[Vec3], list[Vec3], list[Vec2]):
    unique_positions: list[Vec3] = []
    unique_normals: list[Vec3] = []
    unique_tex_uvs: list[Vec2] = []

    positions: list[Vec3] = []
    normals: list[Vec3] = []
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
                position: Vec3 = Vec3(float(items[1]), float(
                    items[2]), -float(items[3]))
                unique_positions.append(position)
            elif (id == "vn"):
                # -z to do right-handed to left-handed coord system change
                normal: Vec3 = Vec3(float(items[1]), float(
                    items[2]), -float(items[3]))
                unique_normals.append(normal)
            elif (id == "vt"):
                tex_uv: Vec2 = Vec2(float(items[1]), float(items[2]))
                unique_tex_uvs.append(tex_uv)
            elif (id == "mtllib"):
                pass

            elif (id == "f"):
                atribs: list[list[int]] = [
                    [int(attribute_index) - 1 for attribute_index in vertex.split("/")] for vertex in items[1:]]
                positions.append(unique_positions[atribs[0][0]])
                positions.append(unique_positions[atribs[1][0]])
                positions.append(unique_positions[atribs[2][0]])

                normals.append(unique_normals[atribs[0][2]])
                normals.append(unique_normals[atribs[1][2]])
                normals.append(unique_normals[atribs[2][2]])

                tex_uvs.append(unique_tex_uvs[atribs[0][1]])
                tex_uvs.append(unique_tex_uvs[atribs[1][1]])
                tex_uvs.append(unique_tex_uvs[atribs[2][1]])

    return (positions, normals, tex_uvs)
