from functools import lru_cache
from typing import List

import FreeCAD as App
import Part

LENGTH_LEDGE = 2
HEIGHT_LEDGE = 2


def _get_points(z_level: float, bridge_width: float) -> List[App.Vector]:
    w = bridge_width / 2
    length = w / 2
    return [
        App.Vector(-(w + LENGTH_LEDGE), 0, z_level),
        App.Vector(-0.25 * w, 0, z_level),
        App.Vector(-0.75 * w, length, z_level),
        App.Vector(0.75 * w, length, z_level),
        App.Vector(0.25 * w, 0, z_level),
        App.Vector((w + LENGTH_LEDGE), 0, z_level),
    ]

@lru_cache(maxsize=None)
def generate_dovetail(bridge_height: float, bridge_width: float) -> Part.Shape:
    wire1 = Part.makePolygon(_get_points(-HEIGHT_LEDGE, bridge_width))
    wire2 = Part.makePolygon(_get_points(bridge_height + HEIGHT_LEDGE, bridge_width))
    return Part.makeRuledSurface(wire1, wire2)
