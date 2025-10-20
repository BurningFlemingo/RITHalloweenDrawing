from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetManager import *
from Sampling import *
from shaders.Lighting import *
from Rasterizer import *


class GeometryVertexShader:
    class Attributes(NamedTuple):
        pos: Vec3
        tex_uv: Vec2
        
        normal: Vec3
        tangent: Vec3
        bitangent: Vec3

    class OutAttributes(NamedTuple):
        pos: Vec3
        frag_light_space_pos: Vec4
        tex_uv: Vec2
        tbn_matrix: Mat4

    def __init__(self, model_matrix: Mat4, normal_matrix: Mat4, view_matrix: Mat4, projection_matrix: Mat4, light_space_matrix: Mat4):
        self.model_matrix = model_matrix
        self.normal_matrix = normal_matrix
        self.view_matrix = view_matrix
        self.projection_matrix = projection_matrix
        self.light_space_matrix = light_space_matrix

    def __call__(self, in_attributes: Attributes) -> Vertex:
        model_matrix: Mat4 = self.model_matrix
        normal_matrix: Mat4 = self.normal_matrix
        view_matrix: Mat4 = self.view_matrix
        projection_matrix: Mat4 = self.projection_matrix
        light_space_matrix: Mat4 = self.light_space_matrix

        pos, tex_uv, normal, tangent, bitangent = in_attributes

        world_pos: Vec4 = model_matrix * Vec4(*pos, 1.0)
        view_pos: Vec4 = view_matrix * world_pos

        model_view_matrix: Mat4 = view_matrix * normal_matrix
        
        T: Vec3 = normalize(model_view_matrix * Vec4(*tangent, 0.0))
        B: Vec3 = normalize(model_view_matrix * Vec4(*bitangent, 0.0))
        N: Vec3 = normalize(model_view_matrix * Vec4(*normal, 0.0))

        tbn_matrix: Mat4 = Mat4(
            Vec4(T.x, B.x, N.x, 0.0),
            Vec4(T.y, B.y, N.y, 0.0),
            Vec4(T.z, B.z, N.z, 0.0),
            Vec4(0.0, 0.0, 0.0, 0.0),
        )
        
        frag_light_space_pos: Vec4 = light_space_matrix * world_pos

        out_position = projection_matrix * view_pos
        out_attributes = self.OutAttributes(
            pos=view_pos.xyz, frag_light_space_pos=frag_light_space_pos, tex_uv=tex_uv, tbn_matrix=tbn_matrix)

        return Vertex(pos=out_position, fragment_attributes=out_attributes)


class GeometryFragmentShader:
    def __init__(self, material: Material, skybox: Sampler3D):
        self.material = material
        self.skybox = skybox

    def __call__(self, attributes: GeometryVertexShader.OutAttributes) -> list[Vec4]:
        uv: Vec2 = attributes.tex_uv
        tbn_matrix: Mat4 = attributes.tbn_matrix
        pos: Vec4 = Vec4(*attributes.pos, 1.0)
        
        material: Material = self.material

        specular: Vec3 = material.specular_color * material.specular_map.sample(*uv).xyz
        sharpness: float = specular.magnitude() * material.specular_sharpness
        albedo: Vec3 = material.diffuse_color * material.diffuse_map.sample(*uv).xyz

        
        frag_light_space_pos: Vec4 = attributes.frag_light_space_pos
        frag_light_space_pos /= frag_light_space_pos.w
        
        normal: Vec3 = material.normal_map.sample(*uv).xyz
        normal = (normal * 2) - 1
        normal = (tbn_matrix * Vec4(*normal, 0.0)).xyz
        normal = normalize(normal)
        
        return [pos, frag_light_space_pos, Vec4(*normal, 1.0), Vec4(*albedo, sharpness)]
