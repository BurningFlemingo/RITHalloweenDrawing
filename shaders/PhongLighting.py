from math import nan
from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetManager import *
from Sampling import *
from shaders.Lighting import *
from shaders.Quad import *
from Rasterizer import *


class PhongFragmentShader:
    def __init__(self, positions: Sampler2D, light_positions: Sampler2D, normals: Sampler2D, albedo: Sampler2D, shadow_map: Sampler2D, skybox: Sampler3D, point_lights: list[PointLight], directional_lights: list[DirectionalLight], spot_lights: list[SpotLight]):
        self.positions = positions
        self.light_positions = light_positions
        self.normals = normals
        self.albedo = albedo
        self.shadow_map = shadow_map
        self.skybox = skybox
        self.point_lights = point_lights
        self.directional_lights = directional_lights
        self.spot_lights = spot_lights


    def __call__(self, attributes: QuadVertexShader.OutAttributes) -> list[Vec4]:
        uv: Vec2 = attributes.tex_uv
        shadow_map: Sampler2D = self.shadow_map
        pos_and_stencil: Vec4 = self.positions.sample(*uv)
        if (pos_and_stencil.w == 0):
            return [Vec4(1, 1, 1, 1)]
        
        pos: Vec3 = pos_and_stencil.xyz
        frag_light_space_pos: Vec3 = self.light_positions.sample(*uv).xyz
        normal: Vec3 = self.normals.sample(*uv).xyz
        albedo: Vec4 = self.albedo.sample(*uv)

        view_dir: Vec3 = normalize(pos * -1)
        reflected_view_dir: Vec3 = reflect(view_dir, normal)
        skybox_frag_color: Vec3 = self.skybox.sample(reflected_view_dir).xyz * 5

        shadow_map_uv: Vec2 = Vec2(
            (frag_light_space_pos.x / 2) + 0.5, (frag_light_space_pos.y / 2) + 0.5)

        max_bias: float = 0.002
        min_bias: float = 0.00001
        spot_light_dir: Vec3 = self.spot_lights[0].pos - pos
        bias: float = max(
            max_bias * (1 - dot(normal, spot_light_dir)), min_bias)

        current_depth: float = frag_light_space_pos.z
        shadow_scalar: float = 0.0
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

        frag_color: Vec3 = Vec3(0.0, 0.0, 0.0)
        for light in self.point_lights:
            frag_color += calc_point_light_contribution(
                light, pos, normal, albedo, view_dir)
        for light in self.directional_lights:
            frag_color += calc_directional_light_contribution(
                light, pos, normal, albedo, view_dir)
        for light in self.spot_lights:
            frag_color += calc_spot_light_contribution(
                light, pos, normal, albedo, view_dir, shadow_scalar)
        # frag_color += skybox_frag_color
        
        return [Vec4(*frag_color, 1)]
