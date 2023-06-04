import math
from enum import Enum
from typing import Tuple, List, Dict


from utils.functions import ellipsoid_formula
from utils.models import Ellipsoid, Vector


class ConnectionType(Enum):
    OneCut = 1
    TwoCuts = 2


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
        "P3": ellipsoid_formula(e, Vector(e.r1 * math.cos(math.pi / 3), e.r2 * math.cos(7*math.pi / 24))),
        "F3": ellipsoid_formula(e, Vector(e.r1 * math.cos(math.pi / 3), -e.r2 * math.cos(7*math.pi / 24))),
        "P4": ellipsoid_formula(e, Vector(-e.r1 * math.cos(math.pi / 3), e.r2 * math.cos(7*math.pi / 24))),
        "F4": ellipsoid_formula(e, Vector(-e.r1 * math.cos(math.pi / 3), -e.r2 * math.cos(7*math.pi / 24))),
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
