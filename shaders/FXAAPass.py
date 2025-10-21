from turtle import left
from typing import NamedTuple

from VectorMath import *
from MatrixMath import *
from shaders.Lighting import *
from shaders.Quad import *
from Sampling import *

class FXAAFragmentShader:
    def __init__(self, color_attachment: Sampler2D):
        self.color_attachment = color_attachment
        
    def __call__(self, attributes: QuadVertexShader.OutAttributes) -> list[Vec4]:
        # https://blog.simonrodriguez.fr/articles/2016/07/implementing_fxaa.html
        min_edge_threshold: float = 0.0312
        max_edge_threshold: float = 0.125
        
        uv: Vec2 = attributes.tex_uv
        screen_tex: Sampler2D = self.color_attachment
        screen_size: SizeDimensions = screen_tex.get_size()
        screen_offset: Vec2 = Vec2(1/screen_size.width, 1/screen_size.height)

        center_color: Vec3 = screen_tex.sample(*uv).xyz
        
        center_luma: float = screen_tex.sample(*uv).w
        up_luma: float = screen_tex.sample(uv.x, uv.y + screen_offset.y, WrappingMode.CLAMP).w
        down_luma: float = screen_tex.sample(uv.x, uv.y - screen_offset.y, WrappingMode.CLAMP).w
        left_luma: float = screen_tex.sample(uv.x - screen_offset.x, uv.y, WrappingMode.CLAMP).w
        right_luma: float = screen_tex.sample(uv.x + screen_offset.x, uv.y, WrappingMode.CLAMP).w

        max_luma: float = max(center_luma, max(max(up_luma, down_luma), max(left_luma, right_luma)))
        min_luma: float = min(center_luma, min(min(up_luma, down_luma), min(left_luma, right_luma)))

        luma_contrast: float = max_luma - min_luma
        if (luma_contrast < max(min_edge_threshold, max_edge_threshold * max_luma)):
            return [Vec4(*center_color, 1.0)]

        top_right_luma: float = screen_tex.sample(uv.x + screen_offset.x, uv.y + screen_offset.y, WrappingMode.CLAMP).w
        bottom_right_luma: float = screen_tex.sample(uv.x + screen_offset.x, uv.y - screen_offset.y, WrappingMode.CLAMP).w
        top_left_luma: float = screen_tex.sample(uv.x - screen_offset.x, uv.y + screen_offset.y, WrappingMode.CLAMP).w
        bottom_left_luma: float = screen_tex.sample(uv.x - screen_offset.x, uv.y - screen_offset.y, WrappingMode.CLAMP).w

        horizontal_edge: float = \
            abs((top_left_luma - left_luma) - (left_luma - bottom_left_luma)) + 2 * \
            abs((up_luma - center_luma) - (center_luma - down_luma)) + \
            abs((top_right_luma - right_luma) - (right_luma - bottom_right_luma))

        vertical_edge: float = \
            abs((top_right_luma - up_luma) - (up_luma - top_left_luma)) + 2 * \
            abs((right_luma - center_luma) - (center_luma - left_luma)) + \
            abs((bottom_right_luma - down_luma) - (down_luma - bottom_left_luma))

        edge_is_horizontal: bool = horizontal_edge >= vertical_edge
        if (edge_is_horizontal):
            luma1: float = down_luma
            luma2: float = up_luma
            gradient_step: Vec2 = Vec2(0.0, screen_offset.y)
            edge_step: Vec2 = Vec2(screen_offset.x, 0.0)
        else: 
            luma1: float = left_luma
            luma2: float = right_luma
            gradient_step: Vec2 = Vec2(screen_offset.x, 0.0)
            edge_step: Vec2 = Vec2(0.0, screen_offset.y)

        negative_gradient: float = luma1 - center_luma
        positive_gradient: float = luma2 - center_luma

        normalized_gradient = max(abs(positive_gradient), abs(negative_gradient)) / 4

        if (negative_gradient >= positive_gradient):
            gradient_step *= -1
            local_luma_avg: float = (center_luma + luma1) / 2
        else: 
            local_luma_avg: float = (center_luma + luma2) / 2

        forward_end_uv: Vec2 = uv + edge_step 
        backward_end_uv: Vec2 = uv - edge_step
        dforward_luma: float = 0
        dbackward_luma: float = 0
        
        reached_forward_end: bool = False
        reached_backward_end: bool = False
        reached_both: bool = False
        quality: list[float] = [1, 1, 1, 1, 1, 1.5, 2, 2, 2, 2, 4, 8]
        max_iterations: int = len(quality)
        for i in range(1, max_iterations):
            if (not reached_forward_end):
                dforward_luma = screen_tex.sample(*forward_end_uv, WrappingMode.CLAMP).w - local_luma_avg
                reached_forward_end = abs(dforward_luma) >= normalized_gradient
                
            if (not reached_backward_end):
                dbackward_luma = screen_tex.sample(*backward_end_uv, WrappingMode.CLAMP).w - local_luma_avg
                reached_backward_end = abs(dbackward_luma) >= normalized_gradient

            reached_both = reached_forward_end and reached_backward_end

            if (reached_both):
                break
            
            if (not reached_forward_end):
                forward_end_uv += edge_step * quality[i]
            else:
                backward_end_uv -= edge_step * quality[i]

        forward_distance: float = uv.x - forward_end_uv.x if edge_is_horizontal else uv.y - forward_end_uv.y
        backward_distance: float = backward_end_uv.x - uv.x if edge_is_horizontal else backward_end_uv.y - uv.y
        smaller_distance: float = min(forward_distance, backward_distance)
        
        edge_length: float = forward_distance + backward_distance
        px_offset: float = 0.5 - (smaller_distance / edge_length)
        
        if (forward_distance < backward_distance):
            correct_variation: bool = (dforward_luma < 0) != (center_luma < local_luma_avg)
        else:
            correct_variation: bool = (dbackward_luma < 0) != (center_luma < local_luma_avg)
        
        final_offset: float = px_offset if correct_variation else 0

        luma_avg: float = 1/12 * (2 * (down_luma + up_luma + left_luma + right_luma) + top_left_luma + top_right_luma + bottom_left_luma + bottom_right_luma)
        subpx_offset1: float = min(max(abs(luma_avg - center_luma) / luma_contrast, 0), 1)
        subpx_offset2: float = (-2 * subpx_offset1 + 3) * (subpx_offset1 ** 2)

        subpx_quality: float = 0.75
        final_subpx_offset: float = (subpx_offset2 ** 2) * subpx_quality

        final_offset = max(final_offset, final_subpx_offset)
        final_uv: Vec2 = uv + (gradient_step * final_offset)

        frag_color: Vec4 = screen_tex.sample(*final_uv, WrappingMode.CLAMP)
            

        return [frag_color]
