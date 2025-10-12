from AssetLoader import *


class Material(NamedTuple):
    ambient_color: Vec3
    diffuse_color: Vec3
    specular_color: Vec3

    ambient_map: Buffer
    diffuse_map: Buffer
    specular_map: Buffer

    specular_sharpness: float

class Mesh(NamedTuple):
    material: Material

    positions: list[Vec3]
    normals: list[Vec3]
    tex_uvs: list[Vec2]

    num_vertices: int

class AssetManager:
    def __init__(self):
        self.m_loaded_texture_cache: dict[str, Buffer] = {}

    def load_model(self, model_path: str) -> list[Mesh]:
        mesh_assets: list[MeshAsset] = parse_obj(model_path)
        meshes: list[Mesh] = []

        for mesh_asset in mesh_assets:
            material: Material = mesh_asset.material
            if (material.ambient_map_path not in self.m_loaded_texture_cache):
                self.m_loaded_texture_cache[material.ambient_map_path] = \
                    load_bmp(material.ambient_map_path, is_srgb_nonlinear=True)

            if (material.diffuse_map_path not in self.m_loaded_texture_cache):
                self.m_loaded_texture_cache[material.diffuse_map_path] = \
                    load_bmp(material.diffuse_map_path, is_srgb_nonlinear=True)

            if (material.specular_map_path not in self.m_loaded_texture_cache):
                self.m_loaded_texture_cache[material.specular_map_path] = \
                    load_bmp(material.specular_map_path, is_srgb_nonlinear=True)

            ambient_map: Buffer = self.m_loaded_texture_cache[material.ambient_map_path]
            diffuse_map: Buffer = self.m_loaded_texture_cache[material.diffuse_map_path]
            specular_map: Buffer = self.m_loaded_texture_cache[material.specular_map_path]
            loaded_material: Material = Material(
                material.ambient_color, material.diffuse_color, material.specular_color,
                ambient_map, diffuse_map, specular_map, material.specular_sharpness
            )
            mesh: Mesh = Mesh(
                loaded_material,
                mesh_asset.positions, mesh_asset.normals, mesh_asset.tex_uvs,
                len(mesh_asset.positions)
            )
            meshes.append(mesh)
        return meshes
