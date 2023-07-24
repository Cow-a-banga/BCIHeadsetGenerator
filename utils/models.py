import math
from functools import lru_cache

import numpy as np
from scipy.optimize import newton


class Vector(np.ndarray):
    def __new__(cls, x: float, y: float, z: float = 0):
        return np.array([x, y, z]).view(cls)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    def normalized(self) -> "Vector":
        norm = np.linalg.norm(self)
        if norm != 0:
            normalized = self / norm
            return normalized.view(type(self))
        else:
            return self.copy()

    def distance(self, other: "Vector") -> float:
        return np.linalg.norm(self - other)

    def cross(self, other: "Vector") -> "Vector":
        result = np.cross(self, other)
        return Vector(result[0], result[1], result[2])


class Ellipsoid:
    r1: float
    r2: float
    r3: float

    def __init__(self, r1: float, r2: float, r3: float):
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3

    def get_normal(self, point: Vector) -> Vector:
        return Vector(2 * point.x / self.r1 ** 2, 2 * point.y / self.r2 ** 2, 2 * point.z / self.r3 ** 2).normalized()

    def closest_point_on_ellipsoid(self, point: Vector) -> Vector:
        normal = self.get_normal(point)

        def ellipsoid_for_closet(x, y, z, a, b, c):
            return (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 - 1

        def f(t):
            x = point[0] + t * normal[0]
            y = point[1] + t * normal[1]
            z = point[2] + t * normal[2]
            return ellipsoid_for_closet(x, y, z, self.r1, self.r2, self.r3)

        t = newton(f, 0)
        return point + t * normal

    @staticmethod
    @lru_cache(maxsize=None)
    def get_ellipse_radius(a, b, theta):
        return a*b/math.sqrt((a*math.sin(theta))**2 + (b*math.cos(theta))**2)

    def get_radius12(self, theta):
        return self.get_ellipse_radius(self.r1, self.r2, theta)

    def get_radius13(self, theta):
        return self.get_ellipse_radius(self.r1, self.r3, theta)

    def get_radius23(self, theta):
        return self.get_ellipse_radius(self.r2, self.r3, theta)

