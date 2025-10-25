from AssetLoader import *
from Sampling import *
from Cubemap import *


class Material(NamedTuple):
    ambient_color: Vec3
    diffuse_color: Vec3
    specular_color: Vec3

    ambient_map: Sampler2D
    diffuse_map: Sampler2D 
    specular_map: Sampler2D 
    normal_map: Sampler2D 

    specular_sharpness: float

class Mesh(NamedTuple):
    material: Material

    positions: list[Vec3]
    tex_uvs: list[Vec2]

    normals: list[Vec3]
    tangents: list[Vec3]
    bitangents: list[Vec3]

    num_vertices: int

class AssetManager:
    def __init__(self):
        self.m_loaded_texture_cache: dict[str, Buffer] = {}

    def load_texture(self, path: str, src_color_space: ColorSpace=ColorSpace.SRGB, dst_color_space: ColorSpace=ColorSpace.LINEAR) -> Buffer:
        if (path not in self.m_loaded_texture_cache):
            self.m_loaded_texture_cache[path] = \
                load_bmp(path, src_color_space, dst_color_space)
                
        return self.m_loaded_texture_cache[path]

    def load_model(self, model_path: str) -> list[Mesh]:
        mesh_assets: list[MeshAsset] = parse_obj(model_path)
        meshes: list[Mesh] = []

        for mesh_asset in mesh_assets:
            material: MaterialAsset = mesh_asset.material

            ambient_map: Buffer = self.load_texture(material.ambient_map_path)
            diffuse_map: Buffer = self.load_texture(material.diffuse_map_path)
            specular_map: Buffer = self.load_texture(material.specular_map_path)
            normal_map: Buffer = self.load_texture(material.normal_map_path, ColorSpace.LINEAR)
            
            loaded_material: Material = Material(
                ambient_color=material.ambient_color,
                diffuse_color=material.diffuse_color, 
                specular_color=material.specular_color,
                ambient_map=Sampler2D(buffers=[ambient_map]).generate_mipmaps(),
                diffuse_map=Sampler2D(buffers=[diffuse_map]).generate_mipmaps(),
                specular_map=Sampler2D(buffers=[specular_map]).generate_mipmaps(),
                normal_map=Sampler2D(buffers=[normal_map], min_filtering_method=FilterMethod.NEAREST, mag_filtering_method=FilterMethod.NEAREST),
                specular_sharpness=material.specular_sharpness
            )
            tangents: list[Vec3] = []
            bitangents: list[Vec3] = []
            
            for i in range(0, len(mesh_asset.positions), 3):
                v1: Vec3 = mesh_asset.positions[i]
                v2: Vec3 = mesh_asset.positions[i + 1]
                v3: Vec3 = mesh_asset.positions[i + 2]
                
                uv1: Vec2 = mesh_asset.tex_uvs[i]
                uv2: Vec2 = mesh_asset.tex_uvs[i + 1]
                uv3: Vec2 = mesh_asset.tex_uvs[i + 2]
                
                tangent, bitangent = calc_tangent_space(v1, v2, v3, uv1, uv2, uv3)
                tangents.extend([tangent, tangent, tangent])
                bitangents.extend([bitangent, bitangent, bitangent])

            mesh: Mesh = Mesh(
                material=loaded_material,
                positions=mesh_asset.positions, tex_uvs=mesh_asset.tex_uvs,
                normals=mesh_asset.normals, tangents=tangents, bitangents=bitangents, 
                num_vertices=len(mesh_asset.positions)
            )
            meshes.append(mesh)
            
        return meshes

    def load_cubemap(self, dir_path: str) -> Sampler3D:
        face_strings: list[str] = [
            "right", "left", "top", "bottom", "front", "back"
        ]
        faces: list[Buffer] = []
        for face_string in face_strings:
            path: str = dir_path + face_string + ".bmp"
            face: Buffer = self.load_texture(path)
            faces.append(face)

        cubemap = Cubemap(faces=faces)
        return Sampler3D(cubemap=cubemap)

def calc_tangent_space(p1: Vec3, p2: Vec3, p3: Vec3, uv1: Vec2, uv2: Vec2, uv3: Vec2):
    e1: Vec3 = p2 - p1
    e2: Vec3 = p3 - p1
    duv1: Vec2 = uv2 - uv1
    duv2: Vec2 = uv3 - uv1
    
    f: float = 1 / ((duv1.x * duv2.y) - (duv2.x * duv1.y))

    tangent: Vec3 = Vec3(
        (duv2.y * e1.x) - (duv1.y * e2.x), 
        (duv2.y * e1.y) - (duv1.y * e2.y), 
        (duv2.y * e1.z) - (duv1.y * e2.z), 
    ) * f

    bitangent: Vec3 = Vec3(
        (-duv2.x * e1.x) + (duv1.x * e2.x), 
        (-duv2.x * e1.y) + (duv1.x * e2.y), 
        (-duv2.x * e1.z) + (duv1.x * e2.z), 
    ) * f

    return tangent, bitangent
