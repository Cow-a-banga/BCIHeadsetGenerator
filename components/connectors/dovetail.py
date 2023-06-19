from functools import lru_cache
from typing import List

import FreeCAD as App
import Part

from components.input.models import ConnectorParameters

WIDTH_LEDGE = 2
HEIGHT_LEDGE = 2


def _get_points(z_level: float, bridge_width: float, connector_length: float) -> List[App.Vector]:
    w = bridge_width / 2
    return [
        App.Vector(-(w + WIDTH_LEDGE), 0, z_level),
        App.Vector(-0.4 * w, 0, z_level),
        App.Vector(-0.6 * w, connector_length, z_level),
        App.Vector(0.6 * w, connector_length, z_level),
        App.Vector(0.4 * w, 0, z_level),
        App.Vector((w + WIDTH_LEDGE), 0, z_level),
    ]

@lru_cache(maxsize=None)
def generate_dovetail(bridge_height: float, bridge_width: float, connector_length: float) -> Part.Shape:
    wire1 = Part.makePolygon(_get_points(-HEIGHT_LEDGE, bridge_width, connector_length))
    wire2 = Part.makePolygon(_get_points(bridge_height + HEIGHT_LEDGE, bridge_width, connector_length))
    return Part.makeRuledSurface(wire1, wire2)
