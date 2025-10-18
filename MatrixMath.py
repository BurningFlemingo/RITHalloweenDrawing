from typing import NamedTuple
from RenderTypes import *
from VectorMath import *


class Mat4(NamedTuple):
    row1: Vec4
    row2: Vec4
    row3: Vec4
    row4: Vec4

    def __mul__(self, other):
        if isinstance(other, Mat4):
            transposed = transpose(other)
            rows: list[Vec4] = []
            for row in self:
                rows.append(Vec4(*[dot(row, col) for col in transposed]))
            return Mat4(*rows)
        elif isinstance(other, Vec4):
            return Vec4(*[dot(row, other) for row in self])
        
        rows: list[Vec4] = []
        for row in self:
            rows.append(row * other)
        return Mat4(*rows)

    def __add__(self, other):
        if isinstance(other, Mat4):
            rows: list[Vec4] = [a + b for a, b in zip(self, other)]
            return Mat4(*rows)
        
    def __truediv__(self, other):
        if not isinstance(other, Mat4) and not isinstance(other, Vec4):
            rows: list[Vec4] = []
            for row in self:
                rows.append(row / other)
            return Mat4(*rows)


def det2x2(a: Vec2, b: Vec2) -> float:
    return (a.x * b.y) - (b.x * a.y)


def det3x3(row1: Vec3, row2: Vec3, row3: Vec3) -> float:
    return \
        row1.x * det2x2(Vec2(row2.y, row2.z), Vec2(row3.y, row3.z)) - \
        row1.y * det2x2(Vec2(row2.x, row2.z), Vec2(row3.x, row3.z)) + \
        row1.z * det2x2(Vec2(row2.x, row2.y), Vec2(row3.x, row3.y))


def transpose(mat: Mat4):
    return Mat4(
        Vec4(mat.row1.x, mat.row2.x, mat.row3.x, mat.row4.x),
        Vec4(mat.row1.y, mat.row2.y, mat.row3.y, mat.row4.y),
        Vec4(mat.row1.z, mat.row2.z, mat.row3.z, mat.row4.z),
        Vec4(mat.row1.w, mat.row2.w, mat.row3.w, mat.row4.w))


def make_rotation_matrix(rotor: Rot3) -> Mat4:
    x_basis: Vec3 = rotate_vec(Vec3(1.0, 0.0, 0.0), rotor)
    y_basis: Vec3 = rotate_vec(Vec3(0.0, 1.0, 0.0), rotor)
    z_basis: Vec3 = rotate_vec(Vec3(0.0, 0.0, 1.0), rotor)
    return Mat4(
        Vec4(x_basis.x, y_basis.x, z_basis.x, 0.0),
        Vec4(x_basis.y, y_basis.y, z_basis.y, 0.0),
        Vec4(x_basis.z, y_basis.z, z_basis.z, 0.0),
        Vec4(0.0, 0.0, 0.0, 1.0)
    )


def make_translation_matrix(translation: Vec3) -> Mat4:
    return Mat4(
        Vec4(1.0, 0.0, 0.0, translation.x),
        Vec4(0.0, 1.0, 0.0, translation.y),
        Vec4(0.0, 0.0, 1.0, translation.z),
        Vec4(0.0, 0.0, 0.0, 1.0)
    )


def make_scale_matrix(scalars: Vec3) -> Mat4:
    return Mat4(
        Vec4(scalars.x, 0.0, 0.0, 0),
        Vec4(0.0, scalars.y, 0.0, 0),
        Vec4(0.0, 0.0, scalars.z, 0),
        Vec4(0.0, 0.0, 0.0, 1.0)
    )


def make_lookat_matrix(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    z_basis: Vec3 = normalize(target - eye)
    x_basis: Vec3 = normalize(cross(up, z_basis))
    y_basis: Vec3 = normalize(cross(z_basis, x_basis))
    print(z_basis)
    print(x_basis)
    print(y_basis)
    return Mat4(
        Vec4(x_basis.x, x_basis.y, x_basis.z, -dot(x_basis, eye)),
        Vec4(y_basis.x, y_basis.y, y_basis.z, -dot(y_basis, eye)),
        Vec4(z_basis.x, z_basis.y, z_basis.z, -dot(z_basis, eye)),
        Vec4(0.0, 0.0, 0.0, 1.0),
    )


def make_model_matrix(t: Transform) -> Mat4:
    return make_translation_matrix(t.pos) * make_rotation_matrix(t.rot) * make_scale_matrix(t.scale)


def make_normal_matrix(model_matrix: Mat4) -> Mat4:
    """
        Doesnt work for non-uniform scaled model matrices
    """
    return Mat4(
        Vec4(*model_matrix.row1[:3], 0),
        Vec4(*model_matrix.row2[:3], 0),
        Vec4(*model_matrix.row3[:3], 0),
        Vec4(0, 0, 0, 1),
    )


def make_perspective_matrix(fov: float, ar: float, near_plane: float, far_plane: float) -> Mat4:
    """
        fov in degrees
    """
    return Mat4(
        Vec4(1/(ar * math.tan(math.radians(fov/2))), 0, 0, 0),
        Vec4(0, 1/math.tan(math.radians(fov/2)), 0, 0),
        Vec4(0, 0, far_plane/(far_plane - near_plane), -
             (far_plane * near_plane)/(far_plane - near_plane)),
        Vec4(0, 0, 1, 0))

def make_orthographic_matrix(left: float, right: float, bottom: float, top: float, near: float, far: float) -> Mat4:
    return Mat4(
        Vec4(2/(right - left), 0, 0, -(right + left)/(right - left)),
        Vec4(0, 2/(top - bottom), 0, -(top + bottom)/(top - bottom)),
        Vec4(0, 0, -2 / (far - near), -(far + near)/(far - near)),
        Vec4(0, 0, 0, 1))
