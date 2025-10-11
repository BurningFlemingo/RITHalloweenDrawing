from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetLoader import *


class FragmentAttributes(NamedTuple):
    pos: Vec3
    normal: Vec3
    tex_uv: Vec2


class FragmentUniforms(NamedTuple):
    material: Material
    point_lights: list[PointLight]
    directional_lights: list[DirectionalLight]
    spot_lights: list[SpotLight]


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

def calc_spot_light_contribution(light: SpotLight, fragment_pos: Vec3, normal: Vec3, tex_uv: Vec2, material: Material, view_dir: Vec3) -> Vec3:
        spot_dir = normalize(light.dir)
        light_dir = light.pos - fragment_pos
        light_distance = light_dir.magnitude()
        light_dir = normalize(light_dir)


        ambient: Vec3 = hadamard(
            material.ambient_color, material.diffuse_map.sampleUV(*tex_uv)) * 0.05

        cos_light_dir: float = dot(light_dir * -1, spot_dir)
        intensity: float = (cos_light_dir - light.cos_outer_cutoff) / (light.cos_inner_cutoff - light.cos_outer_cutoff)
        spot_intensity = max(min(intensity, 1.0), 0.0)
        
        attenuation: float = (light.intensity) / (1 + light_distance ** 2)
        
        diffuse_strength: float = max(dot(light_dir, normal), 0)
        diffuse = hadamard(
            material.diffuse_color, material.diffuse_map.sampleUV(*tex_uv)) * diffuse_strength * spot_intensity

        halfway: Vec3 = normalize(view_dir + light_dir)
        spec: float = max(dot(normal, halfway), 0) ** material.specular_sharpness
        specular = hadamard(
            material.specular_color, material.specular_map.sampleUV(*tex_uv)) * spec * spot_intensity
        
        return hadamard(light.color, ambient + diffuse + specular) * attenuation

def fragment_shader(uniforms: FragmentUniforms, attributes: FragmentAttributes) -> Vec4:
    material, point_lights, directional_lights, spot_lights = uniforms
    pos, normal, tex_uv = attributes
    
    normal = normalize(normal)
    view_dir: Vec3 = normalize(pos * -1)

    final_color: Vec3 = Vec3(0.0, 0.0, 0.0)
    for light in point_lights:
        final_color += calc_point_light_contribution(light, pos, normal, tex_uv, material, view_dir)
    for light in directional_lights:
        final_color += calc_directional_light_contribution(light, pos, normal, tex_uv, material, view_dir)
    for light in spot_lights:
        final_color += calc_spot_light_contribution(light, pos, normal, tex_uv, material, view_dir)

    return Vec4(*final_color, 1.0)
