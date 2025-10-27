from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetManager import *


def calc_point_light_contribution(light: PointLight, fragment_pos: Vec3, normal: Vec3, tex_uv: Vec2, material: Material, view_dir: Vec3, shadow_scalar: float = 1.0, occlusion: float = 1.0) -> Vec3:
    light_dir: Vec3 = light.pos - fragment_pos
    light_distance: float = light_dir.magnitude()
    light_dir = normalize(light_dir)

    attenuation: float = light.intensity / (1 + light_distance ** 2)
    intensity: float = light.intensity * shadow_scalar
    return calc_phong_lighting(material, tex_uv, light.color, light_dir, view_dir, normal, attenuation, intensity, occlusion, light.intensity)


def calc_directional_light_contribution(light: DirectionalLight, fragment_pos: Vec3, normal: Vec3, tex_uv: Vec2, material: Material, view_dir: Vec3, shadow_scalar: float = 1.0, occlusion: float = 1.0) -> Vec3:
    light_dir = normalize(light.dir * -1)

    intensity: float = light.intensity * shadow_scalar
    return calc_phong_lighting(material, tex_uv, light.color, light_dir, view_dir, normal, 1.0, intensity, occlusion, light.intensity)


def calc_spot_light_contribution(light: SpotLight, fragment_pos: Vec3, normal: Vec3, tex_uv: Vec2, material: Material, view_dir: Vec3, shadow_scalar: float = 1.0, occlusion: float = 1.0) -> Vec3:
    spot_dir = normalize(light.dir)
    light_dir = light.pos - fragment_pos
    light_distance = light_dir.magnitude()
    light_dir = normalize(light_dir)

    cos_light_dir: float = dot(light_dir * -1, spot_dir)
    intensity: float = (cos_light_dir - light.cos_outer_cutoff) / \
        (light.cos_inner_cutoff - light.cos_outer_cutoff)

    spot_intensity: float = max(min(intensity, 1.0), 0.0)
    intensity = spot_intensity * shadow_scalar
    attenuation: float = (light.intensity) / (1 + light_distance ** 2)

    return calc_phong_lighting(material, tex_uv, light.color, light_dir, view_dir, normal, attenuation, intensity, occlusion, light.intensity)


def calc_phong_lighting(material: Material, tex_uv: Vec2, light_color: Vec3, light_dir: Vec3, view_dir: Vec3, normal: Vec3, attenuation: float, intensity: float, occlusion: float = 1.0, ambient_intensity: float = 1.0):
    ambient: Vec3 = material.ambient_color * \
        material.diffuse_map.sample(*tex_uv).xyz * ambient_intensity * occlusion
    diffuse: Vec3 = material.diffuse_color * \
        material.diffuse_map.sample(*tex_uv).xyz
    specular: Vec3 = material.specular_color * \
        material.specular_map.sample(*tex_uv).xyz

    diffuse_strength: float = max(dot(light_dir, normal), 0)
    diffuse *= diffuse_strength * intensity

    halfway: Vec3 = normalize(view_dir + light_dir)
    spec: float = max(dot(normal, halfway), 0) ** material.specular_sharpness
    specular *= spec * intensity

    return light_color * (ambient + diffuse + specular) * attenuation 
