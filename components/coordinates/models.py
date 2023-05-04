import math
from enum import Enum
from typing import Tuple, List, Dict


from utils.functions import ellipsoid_formula
from utils.models import Ellipsoid, Vector


class ConnectionType(Enum):
    OneCut = 1
    TwoCuts = 2


#TODO: Ниже
"""
Ваш код вычисляет координаты точек на эллипсоиде с помощью различных формул и углов. Это может быть неэффективно 
с точки зрения производительности и точности, так как вычисления с плавающей точкой могут быть неточными или занимать 
много времени. Я рекомендую вам использовать библиотеку numpy, которая предоставляет оптимизированные функции для работы 
с массивами и матрицами. Вы можете создать массив координат точек на эллипсоиде с помощью функции numpy.meshgrid, а затем 
применить к нему формулу эллипсоида с помощью функции numpy.vectorize. Это должно ускорить ваш код и повысить его точность.

    # Create an array of angles from -pi to pi with a step of pi/5
    angles = np.linspace(-np.pi, np.pi, 11)
    # Create a meshgrid of x and y coordinates from the angles
    x, y = e.r1 * np.cos(angles), e.r2 * np.sin(angles)
    # Create a vectorized function to apply the ellipsoid formula to the meshgrid
    vec_ellipsoid_formula = np.vectorize(ellipsoid_formula)
    # Calculate the z coordinates from the x and y coordinates
    z = vec_ellipsoid_formula(e, x, y)
    # Create a dictionary with the coordinates of the points
    
            return {
            "Cz": Vector(0, 0, e.r3),
            "T3": Vector(e.r1, 0, 0),
            "T4": Vector(-e.r1, 0, 0),
            "C3": Vector(x[2], y[2], z[2]),
            "C4": Vector(x[8], y[8], z[8]),
            "Pz": Vector(0, y[5], z[5]),
            "Fz": Vector(0, y[6], z[6]),
            "O1": Vector(x[9], y[9], 0),
            "O2": Vector(x[10], y[10], 0),
            "Fp1": Vector(x[1], y[1], 0),
            "Fp2": Vector(x[2], y[2], 0),
            "T5": Vector(x[7], y[7], 0),
            "T6": Vector(x[8], y[8], 0),
            "F7": Vector(x[3], y[3], 0),
            "F8": Vector(x[4], y[4], 0),
            "P3": Vector(x[5], y[5], z[5]),
            "F3": Vector(x[6], y[6], z[6]),
            "P4": Vector(x[7], y[7], z[7]),
            "F4": Vector(x[8], y[8], z[8]),
        }
"""


def get_coordinates(e: Ellipsoid) -> Dict[str, Vector]:
    return {
        "Cz": Vector(0, 0, e.r3),
        "T3": Vector(e.r1, 0, 0),
        "T4": Vector(-e.r1, 0, 0),
        "C3": Vector(e.r1 * math.cos(math.pi / 4), 0, e.r3 * math.sin(math.pi / 4)),
        "C4": Vector(-e.r1 * math.cos(math.pi / 4), 0, e.r3 * math.sin(math.pi / 4)),
        "Pz": Vector(0, e.r2 * math.cos(math.pi / 4), e.r3 * math.sin(math.pi / 4)),
        "Fz": Vector(0, -e.r2 * math.cos(math.pi / 4), e.r3 * math.sin(math.pi / 4)),
        "O1": Vector(-e.r1 * math.cos(-3 * math.pi / 5), -e.r2 * math.sin(-3 * math.pi / 5), 0),
        "O2": Vector(-e.r1 * math.cos(-2 * math.pi / 5), -e.r2 * math.sin(-2 * math.pi / 5), 0),
        "Fp1": Vector(-e.r1 * math.cos(3 * math.pi / 5), -e.r2 * math.sin(3 * math.pi / 5), 0),
        "Fp2": Vector(-e.r1 * math.cos(2 * math.pi / 5), -e.r2 * math.sin(2 * math.pi / 5), 0),
        "T5": Vector(-e.r1 * math.cos(-4 * math.pi / 5), -e.r2 * math.sin(-4 * math.pi / 5), 0),
        "T6": Vector(-e.r1 * math.cos(-math.pi / 5), -e.r2 * math.sin(-math.pi / 5), 0),
        "F7": Vector(-e.r1 * math.cos(4 * math.pi / 5), -e.r2 * math.sin(4 * math.pi / 5), 0),
        "F8": Vector(-e.r1 * math.cos(math.pi / 5), -e.r2 * math.sin(math.pi / 5), 0),
        "P3": ellipsoid_formula(e, Vector(-e.r1 * math.cos(3 * math.pi / 5), -e.r1 * math.cos(4 * math.pi / 5))),
        "F3": ellipsoid_formula(e, Vector(-e.r1 * math.cos(3 * math.pi / 5), e.r1 * math.cos(4 * math.pi / 5))),
        "P4": ellipsoid_formula(e, Vector(e.r1 * math.cos(3 * math.pi / 5), -e.r1 * math.cos(4 * math.pi / 5))),
        "F4": ellipsoid_formula(e, Vector(e.r1 * math.cos(3 * math.pi / 5), e.r1 * math.cos(4 * math.pi / 5))),
    }


def get_connected_points() -> List[Tuple[str, str, ConnectionType]]:
    return [
        ("Fp1", "Fp2", ConnectionType.TwoCuts),
        ("Fp1", "F7", ConnectionType.TwoCuts),
        ("F7", "T3", ConnectionType.TwoCuts),
        ("T3", "T5", ConnectionType.TwoCuts),
        ("T5", "O1", ConnectionType.TwoCuts),
        ("Fp2", "F8", ConnectionType.TwoCuts),
        ("F8", "T4", ConnectionType.TwoCuts),
        ("T4", "T6", ConnectionType.TwoCuts),
        ("T6", "O2", ConnectionType.TwoCuts),
        ("O1", "O2", ConnectionType.TwoCuts),
        ("Fp1", "F3", ConnectionType.TwoCuts),
        ("F7", "F3", ConnectionType.TwoCuts),
        ("Fz", "F3", ConnectionType.OneCut),
        ("C3", "F3", ConnectionType.TwoCuts),
        ("T3", "C3", ConnectionType.TwoCuts),
        ("Cz", "C3", ConnectionType.TwoCuts),
        ("P3", "C3", ConnectionType.TwoCuts),
        ("T5", "P3", ConnectionType.TwoCuts),
        ("Pz", "P3", ConnectionType.OneCut),
        ("O1", "P3", ConnectionType.TwoCuts),
        ("Fp2", "F4", ConnectionType.TwoCuts),
        ("F8", "F4", ConnectionType.TwoCuts),
        ("Fz", "F4", ConnectionType.OneCut),
        ("C4", "F4", ConnectionType.TwoCuts),
        ("T4", "C4", ConnectionType.TwoCuts),
        ("Cz", "C4", ConnectionType.TwoCuts),
        ("P4", "C4", ConnectionType.TwoCuts),
        ("T6", "P4", ConnectionType.TwoCuts),
        ("Pz", "P4", ConnectionType.OneCut),
        ("O2", "P4", ConnectionType.TwoCuts),
        ("Fz", "Cz", ConnectionType.TwoCuts),
        ("Pz", "Cz", ConnectionType.TwoCuts)
    ]
