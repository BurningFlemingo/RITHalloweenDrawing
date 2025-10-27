from Buffer import *
from VectorMath import *
from dataclasses import dataclass, field


@dataclass
class MaterialAsset:
    ambient_color: Vec3 = Vec3(0.1, 0.1, 0.1)
    diffuse_color: Vec3 = Vec3(1.0, 1.0, 1.0)
    specular_color: Vec3 = Vec3(0.5, 0.5, 0.5)

    ambient_map_path: str = "assets\\defaults\\solid_white.bmp"
    diffuse_map_path: str = "assets\\defaults\\solid_white.bmp"
    specular_map_path: str = "assets\\defaults\\solid_white.bmp"
    normal_map_path: str = "assets\\defaults\\normal_map.bmp"

    specular_sharpness: float = 32
    name: str = "default"


@dataclass
class MeshAsset:
    material: MaterialAsset = field(default_factory=MaterialAsset)

    positions: list[Vec3] = field(default_factory=list)
    normals: list[Vec3] = field(default_factory=list)
    tex_uvs: list[Vec2] = field(default_factory=list)


def load_bmp(path: str, src_color_space: ColorSpace, dst_color_space: ColorSpace) -> Buffer:
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

        channel_max: int = ((2 ** (bytes_per_color_channel * 8)) - 1)
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

                normalized_color: Vec4 = Vec4(
                    (r * channel_inv_max),
                    (g * channel_inv_max),
                    (b * channel_inv_max),
                    a * channel_inv_max
                )
                normalized_color = transfer_color_space(
                    normalized_color, src_color_space, dst_color_space)
                pixels[row_offset + column] = normalized_color

            if row % 100 == 0:
                print("Read row", row, "of", path)

        return Buffer(
            data=pixels, width=width, height=height, n_samples_per_axis=1,
            format=Format.RGBA_SFLOAT, color_space=dst_color_space)


def parse_mtl(path: str) -> dict[str, MaterialAsset]:
    directory: str = ""
    for i in range(len(path) - 1, 0, -1):
        if (path[i] == '\\'):
            directory = path[:i] + "\\"
            break

    materials: dict[str, MaterialAsset] = {}
    with open(path) as mtl:
        material: MaterialAsset = MaterialAsset()

        for line in mtl:
            items: list[str] = line.strip().split()
            if (len(items) == 0):
                continue
            id: str = items[0]

            if (id == '#'):
                continue
            elif (id == 'newmtl'):
                if (material.name != "default"):
                    materials[material.name] = material
                material = MaterialAsset(name=items[1])
            elif (id == "Ns"):
                material.specular_sharpness = float(items[1])
            elif (id == "Ka"):
                material.ambient_color = Vec3(
                    float(items[1]), float(items[2]), float(items[3]))
            elif (id == "Kd"):
                material.diffuse_color = Vec3(
                    float(items[1]), float(items[2]), float(items[3]))
            elif (id == "Ks"):
                material.specular_color = Vec3(
                    float(items[1]), float(items[2]), float(items[3]))
            elif (id == "map_Ka"):
                material.ambient_map_path = directory + items[1]
            elif (id == "map_Kd"):
                material.diffuse_map_path = directory + items[1]
            elif (id == "map_Ks"):
                material.specular_map_path = directory + items[1]
            elif (id == "bump"):
                material.normal_map_path = directory + items[1]

        materials[material.name] = material

    [print(materials[key].name, ":", materials[key].diffuse_map_path)
     for key in materials]
    return materials


def parse_obj(path: str) -> list[MeshAsset]:
    unique_positions: list[Vec3] = []
    unique_normals: list[Vec3] = []
    unique_tex_uvs: list[Vec2] = []

    materials: dict[str, MaterialAsset] = {}
    meshes: list[MeshAsset] = []

    directory: str = ""
    for i in range(len(path) - 1, 0, -1):
        if (path[i] == '\\'):
            directory = path[:i] + "\\"
            break

    with open(path) as obj:
        current_mesh: MeshAsset = MeshAsset()
        for line in obj:
            items: list[str] = line.strip().split()
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
                materials.update(parse_mtl(directory + items[1]))
            elif (id == "usemtl"):
                if (len(current_mesh.positions) != 0):
                    meshes.append(current_mesh)
                current_mesh = MeshAsset(material=materials[items[1]])
            elif (id == "f"):
                attribs: list[list[int]] = []
                for vertex in items[1:]:
                    vertex_attribute_indices: list[int] = []
                    for attribute_index_str in vertex.split('/'):
                        if (attribute_index_str == ""):
                            vertex_attribute_indices.append(0)
                        else:
                            vertex_attribute_indices.append(
                                int(attribute_index_str) - 1)
                    attribs.append(vertex_attribute_indices)

                current_mesh.positions.append(unique_positions[attribs[0][0]])
                current_mesh.positions.append(unique_positions[attribs[1][0]])
                current_mesh.positions.append(unique_positions[attribs[2][0]])

                current_mesh.normals.append(unique_normals[attribs[0][2]])
                current_mesh.normals.append(unique_normals[attribs[1][2]])
                current_mesh.normals.append(unique_normals[attribs[2][2]])

                current_mesh.tex_uvs.append(unique_tex_uvs[attribs[0][1]])
                current_mesh.tex_uvs.append(unique_tex_uvs[attribs[1][1]])
                current_mesh.tex_uvs.append(unique_tex_uvs[attribs[2][1]])

        meshes.append(current_mesh)

    return meshes
