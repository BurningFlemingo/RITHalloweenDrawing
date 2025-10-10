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
    light: PointLight


def fragment_shader(uniforms: FragmentUniforms, attributes: FragmentAttributes) -> Vec4:
    material, light = uniforms
    pos, normal, tex_uv = attributes

    norm: Vec3 = normalize(normal)

    view_dir: Vec3 = normalize(pos * -1)

    light_dir: Vec3 = light.pos - pos
    light_distance: float = light_dir.magnitude()
    light_dir = normalize(light_dir)

    attenuation: float = 1.0 / \
        (1.0 + light.linear_att * light_distance +
         (light.quadratic_att * light_distance ** 2))

    ambient: Vec3 = hadamard(hadamard(
        light.color, material.diffuse_map.sampleUV(*tex_uv) * 0.2), material.ambient_color)

    diffuse_strength: float = max(dot(light_dir, norm), 0)
    diffuse: Vec3 = hadamard(hadamard(
        light.color, material.diffuse_map.sampleUV(*tex_uv) * diffuse_strength), material.diffuse_color)

    halfway: Vec3 = normalize(view_dir + light_dir)
    spec: float = max(dot(norm, halfway), 0) ** material.specular_sharpness
    specular = hadamard(hadamard(
        light.specular, (material.specular_map.sampleUV(*tex_uv) * spec)), material.specular_color)

    result: Vec3 = (ambient + diffuse + specular) * attenuation

    return Vec4(*result, 1.0)
