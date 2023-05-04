import Draft
import Part
import FreeCAD as App

from typing import Dict, List, Tuple

from components.bridges.curve import get_curves
from components.bridges.models import Connection
from components.coordinates.models import get_connected_points
from components.input.models import InputParameters
from utils.models import Vector


def _render_bridge(params: InputParameters, curve: Draft.BezCurve, point: Vector, doc: App.Document) -> Part.Shape:
    curve_offset = params.bridge.width / 2
    normal: Vector = params.ellipsoid.get_normal(point) * params.bridge.height
    feature = doc.addObject("Part::Feature", "Wire")
    feature.Shape = curve.Shape
    cloned_object = doc.copyObject(feature)
    cloned_object.Placement.Base += App.Vector(normal)

    surface = Part.makeRuledSurface(curve.Shape, cloned_object.Shape)
    offset0 = surface.makeOffsetShape(curve_offset, 0.1, fill=False)
    offset = offset0.makeOffsetShape(-2 * curve_offset, 0.1, fill=True)
    Part.show(offset)
    doc.removeObject(curve.Name)
    doc.removeObject(feature.Name)
    doc.removeObject(cloned_object.Name)
    return offset


def render_bridges(params: InputParameters, points_coordinates: Dict[str, Vector],
                   doc: App.Document) -> Tuple[List[Part.Shape], List[Connection]]:
    connected_points = get_connected_points()
    curves, connections = get_curves(params, points_coordinates, connected_points)
    models: List[Part.Shape] = []

    for curve, connection in zip(curves, connections):
        point = connection.points[0].point
        model = _render_bridge(params, curve, point, doc)
        models.append(model)

    return models, connections
