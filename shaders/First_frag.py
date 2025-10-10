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


def fragment_shader(uniforms: FragmentUniforms, attributes: FragmentAttributes) -> Vec4:
    material, point_lights = uniforms
    pos, normal, tex_uv = attributes

    norm: Vec3 = normalize(normal)

    final_color: Vec3 = Vec3(0.0, 0.0, 0.0)
    for light in point_lights:
        view_dir: Vec3 = normalize(pos * -1)

        light_dir: Vec3 = light.pos - pos
        light_distance: float = light_dir.magnitude()
        light_dir = normalize(light_dir)

        attenuation: float = 1.0 / (light_distance ** 2)

        ambient: Vec3 = hadamard(hadamard(
            light.color, material.diffuse_map.sampleUV(*tex_uv)), material.ambient_color)

        diffuse_strength: float = max(dot(light_dir, norm), 0)
        diffuse: Vec3 = hadamard(hadamard(
            light.color, material.diffuse_map.sampleUV(*tex_uv)), material.diffuse_color) * diffuse_strength

        halfway: Vec3 = normalize(view_dir + light_dir)
        spec: float = max(dot(norm, halfway), 0) ** material.specular_sharpness
        specular = hadamard(hadamard(
            light.specular, material.specular_map.sampleUV(*tex_uv)), material.specular_color) * spec
        
        final_color = final_color + (ambient + diffuse + specular) * attenuation


    return Vec4(*final_color, 1.0)
