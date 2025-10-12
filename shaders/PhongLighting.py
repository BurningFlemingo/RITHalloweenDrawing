from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetLoader import *


class PhongVertexShader:
    class Uniforms(NamedTuple):
        model_matrix: Mat4
        normal_matrix: Mat4
        view_matrix: Mat4
        projection_matrix: Mat4

        light_space_matrix: Mat4

    class Attributes(NamedTuple):
        pos: Vec3
        normal: Vec3
        tex_uv: Vec2

    class OutAttributes(NamedTuple):
        pos: Vec3
        normal: Vec3
        tex_uv: Vec2
        frag_light_space_pos: Vec4

    def __init__(self, uniforms: Uniforms):
        self.uniforms = uniforms

    def __call__(self, in_attributes: Attributes) -> Vertex:
        model_matrix, normal_matrix, view_matrix, projection_matrix, light_space_matrix = self.uniforms
        pos, normal, tex_uv = in_attributes

        view_pos: Vec4 = view_matrix * \
            model_matrix * Vec4(*pos, 1.0)

        normal: Vec3 = Vec3(*(normal_matrix * Vec4(*normal, 1.0))[:3])
        frag_light_space_pos: Vec4 = light_space_matrix * \
            model_matrix * Vec4(*pos, 1.0)

        out_position = projection_matrix * view_pos
        out_attributes = self.OutAttributes(
            pos=Vec3(*view_pos[:3]), normal=normal, tex_uv=tex_uv, frag_light_space_pos=frag_light_space_pos)

        return Vertex(pos=out_position, fragment_attributes=out_attributes)


class PhongFragmentShader:
    class Uniforms(NamedTuple):
        material: Material
        point_lights: list[PointLight]
        directional_lights: list[DirectionalLight]
        spot_lights: list[SpotLight]

        shadow_map: Buffer

    class Attributes(NamedTuple):
        pos: Vec3
        normal: Vec3
        tex_uv: Vec2
        frag_light_space_pos: Vec4

    def __init__(self, uniforms: Uniforms):
        self.uniforms = uniforms

    def __call__(self, attributes: Attributes) -> list[Vec4]:
        uniforms: Uniforms = self.uniforms

        material: Material = uniforms.material
        shadow_map: Buffer = uniforms.shadow_map

        pos: Vec3 = attributes.pos
        normal: Vec3 = attributes.normal
        tex_uv: Vec2 = attributes.tex_uv

        frag_light_space_pos: Vec4 = attributes.frag_light_space_pos
        frag_light_space_pos /= frag_light_space_pos.w

        shadow_map_uv: Vec2 = Vec2(
            (frag_light_space_pos.x / 2) + 0.5, (frag_light_space_pos.y / 2) + 0.5)

        bias: float = 0.0001
        current_depth: float = frag_light_space_pos.z
        
        frag_in_shadow: bool = False
        if(shadow_map_uv.x <= 1.0 and shadow_map_uv.x >= 0 and shadow_map_uv.y <= 1.0 and shadow_map_uv.y >= 0):
            closest_depth: float = shadow_map.sampleUV(*shadow_map_uv)
            
            frag_in_shadow = (current_depth - bias) > closest_depth


        normal = normalize(normal)
        view_dir: Vec3 = normalize(pos * -1)

        final_color: Vec3 = Vec3(0.0, 0.0, 0.0)
        for light in uniforms.point_lights:
            final_color += calc_point_light_contribution(
                light, pos, normal, tex_uv, material, view_dir)
        for light in uniforms.directional_lights:
            final_color += calc_directional_light_contribution(
                light, pos, normal, tex_uv, material, view_dir)
        for light in uniforms.spot_lights:
            final_color += calc_spot_light_contribution(
                light, pos, normal, tex_uv, material, view_dir, frag_in_shadow)

        return [Vec4(*final_color, 1.0)]


def calc_point_light_contribution(light: PointLight, fragment_pos: Vec3, normal: Vec3, tex_uv: Vec2, material: Material, view_dir: Vec3) -> Vec3:
    light_dir: Vec3 = light.pos - fragment_pos
    light_distance: float = light_dir.magnitude()
    light_dir = normalize(light_dir)

    attenuation: float = light.intensity / (1 + light_distance ** 2)

    ambient: Vec3 = hadamard(
        material.ambient_color, material.diffuse_map.sampleUV(*tex_uv))

    diffuse_strength: float = max(dot(light_dir, normal), 0)
    diffuse: Vec3 = hadamard(
        material.diffuse_color, material.diffuse_map.sampleUV(*tex_uv)) * diffuse_strength

    halfway: Vec3 = normalize(view_dir + light_dir)
    spec: float = max(dot(normal, halfway), 0) ** material.specular_sharpness
    specular: Vec3 = hadamard(
        material.specular_color, material.specular_map.sampleUV(*tex_uv)) * spec

    return hadamard(light.color, ambient + diffuse + specular) * attenuation


def calc_directional_light_contribution(light: DirectionalLight, fragment_pos: Vec3, normal: Vec3, tex_uv: Vec2, material: Material, view_dir: Vec3) -> Vec3:
    light_dir = normalize(light.dir * -1)

    ambient: Vec3 = hadamard(
        material.ambient_color, material.diffuse_map.sampleUV(*tex_uv))

    diffuse_strength: float = max(dot(light_dir, normal), 0)
    diffuse: Vec3 = hadamard(
        material.diffuse_color, material.diffuse_map.sampleUV(*tex_uv)) * diffuse_strength

    halfway: Vec3 = normalize(view_dir + light_dir)
    spec: float = max(dot(normal, halfway), 0) ** material.specular_sharpness
    specular: Vec3 = hadamard(
        material.specular_color, material.specular_map.sampleUV(*tex_uv)) * spec

    return hadamard(light.color, ambient + diffuse + specular) * light.intensity


def calc_spot_light_contribution(light: SpotLight, fragment_pos: Vec3, normal: Vec3, tex_uv: Vec2, material: Material, view_dir: Vec3, frag_in_shadow: bool) -> Vec3:
    spot_dir = normalize(light.dir)
    light_dir = light.pos - fragment_pos
    light_distance = light_dir.magnitude()
    light_dir = normalize(light_dir)

    shadow_scalar = 0 if frag_in_shadow else 1

    ambient: Vec3 = hadamard(
        material.ambient_color, material.diffuse_map.sampleUV(*tex_uv))

    cos_light_dir: float = dot(light_dir * -1, spot_dir)
    intensity: float = (cos_light_dir - light.cos_outer_cutoff) / \
        (light.cos_inner_cutoff - light.cos_outer_cutoff)
    spot_intensity = max(min(intensity, 1.0), 0.0)

    attenuation: float = (light.intensity) / (1 + light_distance ** 2)

    diffuse_strength: float = max(dot(light_dir, normal), 0)
    diffuse = hadamard(
        material.diffuse_color, material.diffuse_map.sampleUV(*tex_uv)) * diffuse_strength * spot_intensity * shadow_scalar

    halfway: Vec3 = normalize(view_dir + light_dir)
    spec: float = max(dot(normal, halfway), 0) ** material.specular_sharpness
    specular = hadamard(
        material.specular_color, material.specular_map.sampleUV(*tex_uv)) * spec * spot_intensity * shadow_scalar

    return hadamard(light.color, ambient + diffuse + specular) * attenuation
