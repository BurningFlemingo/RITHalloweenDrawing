import math
from typing import NamedTuple
from typing import Any

class Vec4(NamedTuple):
    x: Any
    y: Any
    z: Any
    w: Any

    def __add__(self, other):
        return type(self)(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        return type(self)(*[a - b for a, b in zip(self, other)])

    def __mul__(self, factor):
        return type(self)(*[a * factor for a in self])

    def __truediv__(self, factor):
        return type(self)(*[a / factor for a in self])

    def __floordiv__(self, factor):
        return type(self)(*[a // factor for a in self])

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)


class Vec3(NamedTuple):
    x: Any
    y: Any
    z: Any

    def __add__(self, other):
        return type(self)(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        return type(self)(*[a - b for a, b in zip(self, other)])

    def __mul__(self, factor):
        return type(self)(*[(a * factor) for a in self])

    def __truediv__(self, factor):
        return type(self)(*[a / factor for a in self])

    def __floordiv__(self, factor):
        return type(self)(*[a // factor for a in self])

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


class Vec2(NamedTuple):
    x: Any
    y: Any

    def __add__(self, other):
        return type(self)(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other):
        return type(self)(*[a - b for a, b in zip(self, other)])

    def __mul__(self, factor):
        return type(self)(*[(a * factor) for a in self])

    def __truediv__(self, factor):
        return type(self)(*[a / factor for a in self])

    def __floordiv__(self, factor):
        return type(self)(*[a // factor for a in self])

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)


class Rot3(NamedTuple):
    scalar: float
    xy: float
    yz: float
    zx: float

    def __mul__(self, other):
        return Rot3(
            self.scalar * other.scalar - self.xy * other.xy -
            self.yz * other.yz - self.zx * other.zx,
            self.scalar * other.xy + self.xy * other.scalar -
            self.yz * other.zx + self.zx*other.yz,
            self.scalar * other.yz + self.xy * other.zx +
            self.yz * other.scalar - self.zx * other.xy,
            self.scalar * other.zx - self.xy * other.yz +
            self.yz * other.xy + self.zx * other.scalar
        )


def normalize(vec):
    return vec / vec.magnitude()


def dot(v1, v2):
    accumulated = 0
    for a, b in zip(v1, v2):
        accumulated += a * b
    return accumulated


def cross(a: Vec3, b: Vec3):
    return Vec3(
        a.y * b.z - a.z * b.y,
        -(a.x * b.z - a.z * b.x),
        a.x * b.y - a.y * b.x
    )


def hadamard(v1, v2):
    return type(v1)(*[a * b for a, b in zip(v1, v2)])


def reflect(incoming, normal):
    return incoming - (normal * (2 * dot(normal, incoming)))


def wedge(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(
        (a.x * b.y) - (a.y * b.x),
        (a.y * b.z) - (a.z * b.y),
        (a.z * b.x) - (a.x * b.z)
    )


def make_rotor(from_vec: Vec3, to_vec: Vec3, theta: float = None):
    """
        equations credit: https://jacquesheunis.com/post/rotors/
    """
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)

    if (theta is None):
        # handle when halfway magnitude is zero
        halfway_vec: Vec3 = normalize(from_vec + to_vec)

        scalar_part: Vec3 = dot(from_vec, halfway_vec)
        wedge_part: Vec3 = normalize(wedge(halfway_vec, from_vec))

        return Rot3(scalar_part, *wedge_part)

    scalar_part: float = math.cos(theta / 2)
    wedge_part: Vec3 = normalize(
        wedge(to_vec, from_vec)) * math.sin(theta / 2.0)

    return Rot3(scalar_part, *wedge_part)


def make_euler_rotor(euler_angles: Vec3) -> Rot3:
    """
        x, y, z angles are interpreted in radians.
    """
    return \
        make_rotor(Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), euler_angles.x) * \
        make_rotor(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), euler_angles.y) * \
        make_rotor(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), euler_angles.z)


def reverse(r: Rot3):
    """
        https://jacquesheunis.com/post/rotors/#eqn:geometric-product-complete
    """
    return Rot3(
        r.scalar,
        -r.xy,
        -r.yz,
        -r.zx
    )


def rotate_vec(v: Vec3, r: Rot3) -> Vec3:
    """
        https://jacquesheunis.com/post/rotors/#eqn:geometric-product-complete
    """

    s_x: float = r.scalar * v.x + r.xy * v.y - r.zx * v.z
    s_y: float = r.scalar * v.y - r.xy * v.x + r.yz * v.z
    s_z: float = r.scalar * v.z - r.yz * v.y + r.zx * v.x
    s_xyz: float = r.xy * v.z + r.yz * v.x + r.zx * v.y

    return Vec3(
        s_x * r.scalar + s_y * r.xy + s_xyz * r.yz - s_z * r.zx,
        s_y * r.scalar - s_x * r.xy + s_z * r.yz + s_xyz * r.zx,
        s_z * r.scalar + s_xyz * r.xy - s_y * r.yz + s_x * r.zx
    )

