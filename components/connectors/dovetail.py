from functools import lru_cache
from typing import List

import FreeCAD as App
import Part

WIDTH_LEDGE = 2
HEIGHT_LEDGE = 2


def _get_points(z_level: float, bridge_width: float) -> List[App.Vector]:
    w = bridge_width / 2
    length = bridge_width
    return [
        App.Vector(-(w + WIDTH_LEDGE), 0, z_level),
        App.Vector(-0.4 * w, 0, z_level),
        App.Vector(-0.6 * w, length, z_level),
        App.Vector(0.6 * w, length, z_level),
        App.Vector(0.4 * w, 0, z_level),
        App.Vector((w + WIDTH_LEDGE), 0, z_level),
    ]

@lru_cache(maxsize=None)
def generate_dovetail(bridge_height: float, bridge_width: float) -> Part.Shape:
    wire1 = Part.makePolygon(_get_points(-HEIGHT_LEDGE, bridge_width))
    wire2 = Part.makePolygon(_get_points(bridge_height + HEIGHT_LEDGE, bridge_width))
    return Part.makeRuledSurface(wire1, wire2)
