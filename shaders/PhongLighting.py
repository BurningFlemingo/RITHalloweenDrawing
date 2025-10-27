from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetManager import *
from Sampling import *
from shaders.Lighting import *
from Rasterizer import *


class PhongVertexShader:
    class Attributes(NamedTuple):
        pos: Vec3
        tex_uv: Vec2
        
        normal: Vec3
        tangent: Vec3
        bitangent: Vec3

    class OutAttributes(NamedTuple):
        pos: Vec3
        tex_uv: Vec2
        frag_light_space_pos: Vec4
        
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
            pos=view_pos.xyz, tex_uv=tex_uv, frag_light_space_pos=frag_light_space_pos, tbn_matrix=tbn_matrix)

        return Vertex(pos=out_position, fragment_attributes=out_attributes)


class PhongFragmentShader:
    def __init__(self, material: Material, point_lights: list[PointLight], directional_lights: list[DirectionalLight], spot_lights: list[SpotLight], castable_light: PointLight | SpotLight | DirectionalLight | None, occlusion_map: Sampler2D, shadow_map: Sampler2D, skybox: Sampler3D, projection_matrix: Mat4):
        self.material = material
        self.point_lights = point_lights
        self.directional_lights = directional_lights
        self.spot_lights = spot_lights
        self.castable_light = castable_light
        self.occlusion_map = occlusion_map
        self.shadow_map = shadow_map
        self.skybox = skybox
        self.projection_matrix = projection_matrix

    def __call__(self, attributes: PhongVertexShader.OutAttributes) -> list[Vec4]:
        material: Material = self.material
        occlusion_map: Sampler2D = self.occlusion_map
        shadow_map: Sampler2D = self.shadow_map

        pos: Vec3 = attributes.pos
        tex_uv: Vec2 = attributes.tex_uv
        
        tbn_matrix: Mat4 = attributes.tbn_matrix

        frag_light_space_pos: Vec4 = attributes.frag_light_space_pos
        frag_light_space_pos /= frag_light_space_pos.w
        current_depth: float = frag_light_space_pos.z

        normal: Vec3 = material.normal_map.sample(*tex_uv).xyz
        normal = (normal * 2) - 1
        normal = (tbn_matrix * Vec4(*normal, 0.0)).xyz
        normal = normalize(normal)
        
        view_dir: Vec3 = normalize(pos * -1)

        reflected_view_dir: Vec3 = reflect(view_dir, normal)
        skybox_frag_color: Vec3 = self.skybox.sample(reflected_view_dir).xyz * 5

        projection_matrix: Mat4 = self.projection_matrix 
        occlusion_ndc: Vec4 = projection_matrix * Vec4(*pos, 1.0)
        occlusion_ndc /= occlusion_ndc.w
        occlusion_uv: Vec2 = (occlusion_ndc.xy / 2) + 0.5
        occlusion: float = occlusion_map.sample(*occlusion_uv).x

        frag_color: Vec3 = Vec3(0.0, 0.0, 0.0)
        for light in self.point_lights:
            frag_color += calc_point_light_contribution(
                light, pos, normal, tex_uv, material, view_dir, 1.0, occlusion)
        for light in self.directional_lights:
            frag_color += calc_directional_light_contribution(
                light, pos, normal, tex_uv, material, view_dir, 1.0, occlusion)
        for light in self.spot_lights:
            frag_color += calc_spot_light_contribution(
                light, pos, normal, tex_uv, material, view_dir, 1.0, occlusion)
        
        shadow_map_uv: Vec2 = Vec2(
            (frag_light_space_pos.x / 2) + 0.5, (frag_light_space_pos.y / 2) + 0.5)
        
        
        castable_dir: Vec3 = Vec3()
        if (isinstance(self.castable_light, PointLight) or isinstance(self.castable_light, SpotLight)):
            castable_dir = self.castable_light.pos - pos
        elif(isinstance(self.castable_light, DirectionalLight)):
            castable_dir = self.castable_light.dir
        
        shadow_scalar: float = 1.0
        if (self.castable_light is not None):
            max_bias: float = 0.002
            min_bias: float = 0.00001
            bias: float = max(
                max_bias * (1 - dot(normal, castable_dir)), min_bias)

            for y in range(-1, 2):
                for x in range(-1, 2):
                    u: float = shadow_map_uv.x + x / shadow_map.get_size().width
                    v: float = shadow_map_uv.y + y / shadow_map.get_size().height
                    closest_depth: float = shadow_map.sample(
                        u, v, WrappingMode.CLAMP_TO_BORDER, float("inf")).x
                    if (current_depth <= 1.0):
                        shadow_scalar += 1 if (current_depth -
                                               bias) > closest_depth else 0
            shadow_scalar = 1 - (shadow_scalar / 9)

        if (isinstance(self.castable_light, PointLight)):
            frag_color += calc_point_light_contribution(
                self.castable_light, pos, normal, tex_uv, material, view_dir, shadow_scalar, occlusion)
        elif (isinstance(self.castable_light, DirectionalLight)):
            frag_color += calc_directional_light_contribution(
                self.castable_light, pos, normal, tex_uv, material, view_dir, shadow_scalar, occlusion)
        elif (isinstance(self.castable_light, SpotLight)):
            frag_color += calc_spot_light_contribution(
                self.castable_light, pos, normal, tex_uv, material, view_dir, shadow_scalar, occlusion)
        
        # bloom_color: Vec3 = Vec3(0, 0, 0)
        # if (brightness > 1.0):
        #     bloom_color = frag_color
        return [Vec4(*frag_color, 1.0)]
