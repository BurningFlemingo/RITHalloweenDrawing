from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetManager import *


def calc_point_light_contribution(light: PointLight, fragment_pos: Vec3, normal: Vec3, albedo: Vec4, view_dir: Vec3) -> Vec3:
    light_dir: Vec3 = light.pos - fragment_pos
    light_distance: float = light_dir.magnitude()
    light_dir = normalize(light_dir)

    attenuation: float = light.intensity / (1 + light_distance ** 2)
    return calc_phong_lighting(albedo, light.color, light_dir, view_dir, normal, attenuation, 1.0)


def calc_directional_light_contribution(light: DirectionalLight, fragment_pos: Vec3, normal: Vec3, albedo: Vec4, view_dir: Vec3) -> Vec3:
    light_dir = normalize(light.dir * -1)

    return calc_phong_lighting(albedo, light.color, light_dir, view_dir, normal, 1.0, light.intensity)


def calc_spot_light_contribution(light: SpotLight, fragment_pos: Vec3, normal: Vec3, albedo: Vec4, view_dir: Vec3, shadow_scalar: float) -> Vec3:
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

    return calc_phong_lighting(albedo, light.color, light_dir, view_dir, normal, attenuation, intensity)


def calc_phong_lighting(albedo: Vec4, light_color: Vec3, light_dir: Vec3, view_dir: Vec3, normal: Vec3, attenuation: float, intensity: float):
    ambient: Vec3 = albedo.xyz * 0.2
    diffuse: Vec3 = albedo.xyz
    specular: Vec3 = Vec3(1.0, 1.0, 1.0)
    specular_sharpness: float = albedo.w
    
    diffuse_strength: float = max(dot(light_dir, normal), 0)
    diffuse *= diffuse_strength * intensity

    halfway: Vec3 = normalize(view_dir + light_dir)
    spec: float = max(dot(normal, halfway), 0) ** specular_sharpness
    specular *= spec * intensity

    return light_color * (ambient + diffuse + specular) * attenuation
