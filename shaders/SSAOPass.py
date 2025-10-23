from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from RenderTypes import *
from AssetManager import *
from Sampling import *
from Rasterizer import *


class SSAOVertexShader:
    class Attributes(NamedTuple):
        pos: Vec3
        tex_uv: Vec2
        normal: Vec3
        tangent: Vec3
        bitangent: Vec3

    class OutAttributes(NamedTuple):
        pos: Vec3
        tex_uv: Vec2
        tbn_matrix: Mat4

    def __init__(self, model_matrix: Mat4, normal_matrix: Mat4, view_matrix: Mat4, projection_matrix: Mat4):
        self.model_matrix = model_matrix
        self.normal_matrix = normal_matrix
        self.view_matrix = view_matrix
        self.projection_matrix = projection_matrix

    def __call__(self, in_attributes: Attributes) -> Vertex:
        model_matrix: Mat4 = self.model_matrix
        normal_matrix: Mat4 = self.normal_matrix
        view_matrix: Mat4 = self.view_matrix
        projection_matrix: Mat4 = self.projection_matrix

        pos, tex_uv, normal, tangent, bitangent = in_attributes

        world_pos: Vec4 = model_matrix * Vec4(*pos, 1.0)
        view_pos: Vec4 = view_matrix * world_pos

        model_view_matrix: Mat4 = view_matrix * model_matrix
        
        T: Vec3 = normalize(model_view_matrix * Vec4(*tangent, 0.0))
        B: Vec3 = normalize(model_view_matrix * Vec4(*bitangent, 0.0))
        N: Vec3 = normalize(model_view_matrix * Vec4(*normal, 0.0))

        tbn_matrix: Mat4 = Mat4(
            Vec4(T.x, B.x, N.x, 0.0),
            Vec4(T.y, B.y, N.y, 0.0),
            Vec4(T.z, B.z, N.z, 0.0),
            Vec4(0.0, 0.0, 0.0, 1.0),
        )


        out_position = projection_matrix * view_pos
        out_attributes = self.OutAttributes(
            pos=view_pos.xyz, tex_uv=tex_uv, tbn_matrix=tbn_matrix)

        return Vertex(pos=out_position, fragment_attributes=out_attributes)


class SSAOFragmentShader:
    def __init__(self, projection_matrix: Mat4, normal_map: Sampler2D, pre_pass_depth_buffer: Sampler2D, kernel: list[Vec3], noise: Sampler2D):
        self.projection_matrix = projection_matrix
        self.normal_map = normal_map
        self.pre_pass_depth_buffer = pre_pass_depth_buffer
        self.kernel = kernel
        self.noise = noise

    def __call__(self, attributes: SSAOVertexShader.OutAttributes) -> list[Vec4]:
        prepass_depth_buffer: Sampler2D = self.pre_pass_depth_buffer

        projection_matrix: Mat4 = self.projection_matrix
        
        frag_pos: Vec3 = attributes.pos
        tex_uv: Vec2 = attributes.tex_uv
        screen_size: SizeDimensions = prepass_depth_buffer.get_size()
        normal_sample: Vec4 = self.normal_map.sample(*tex_uv) * 2 - 1
        normal: Vec3 = (attributes.tbn_matrix * normal_sample).xyz
        
        normal = normalize(normal)
        tangent: Vec3 = self.noise.sample(*(tex_uv * Vec2(screen_size.width, screen_size.height)), mode=WrappingMode.REPEAT).xyz
        tangent = normalize(tangent - (normal * dot(tangent, normal)))
        bitangent: Vec3 = normalize(cross(normal, tangent))
        kernel_tbn: Mat4 = Mat4(
            Vec4(tangent.x, bitangent.x, normal.x, 0),
            Vec4(tangent.y, bitangent.y, normal.y, 0),
            Vec4(tangent.z, bitangent.z, normal.z, 0),
            Vec4(0, 0, 0, 1),
        )

        radius: float = 0.5
        occlusion: float = 0
        max_bias: float = 0.0005
        min_bias: float = 0.00003
        frag_dir: Vec3 = -1 * frag_pos
        bias: float = max(
            max_bias * (1 - dot(normal, frag_dir)), min_bias)
        for sample in self.kernel:
            sample_pos: Vec3 = frag_pos + (kernel_tbn * Vec4(*sample, 1.0)).xyz * radius
            sample_ndc: Vec4 = projection_matrix * Vec4(*sample_pos, 1.0)
            sample_ndc /= sample_ndc.w
            sample_uv: Vec2 = (sample_ndc.xy / 2) + 0.5
            
            current_depth: float = prepass_depth_buffer.sample(*sample_uv, mode=WrappingMode.CLAMP).x
            sample_depth: float = sample_ndc.z
            depth_range: float = max(min(radius/(abs(current_depth - sample_depth)), 1), 0)
            range_check: float = (depth_range ** 2) * (3 - (2 * depth_range))
            occlusion += (1 if (current_depth + bias) >= sample_depth else 0) * range_check

        occlusion /= len(self.kernel)
        frag_color: Vec3 = Vec3(occlusion, occlusion, occlusion)
        return [Vec4(*frag_color, 1.0)]
