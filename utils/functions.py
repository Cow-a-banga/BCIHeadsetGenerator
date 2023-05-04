import math

import numpy as np

from utils.models import Vector, Ellipsoid


def rotation_matrix_from_vectors(vec1: Vector, vec2: Vector) -> np.ndarray:
    """
    Возвращает матрицу поворота, которая поворачивает vec1 в vec2
    Использует алгоритм Родрига для вычисления матрицы поворота
    """

    a, b = vec1.normalized(), vec2.normalized()
    v = np.cross(a, b)
    if any(v):
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3)


def ellipsoid_formula(e: Ellipsoid, p: Vector) -> Vector:
    formula = 1 - (p.x/e.r1)**2 - (p.y/e.r2)**2
    if formula < 0:
        return Vector(p.x, p.y, 0)
    return Vector(p.x, p.y, e.r3 * math.sqrt(formula))