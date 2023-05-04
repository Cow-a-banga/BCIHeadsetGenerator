from typing import Tuple, Dict, List

import Draft
import FreeCAD as App

from components.bridges.connection import get_point_on_socket_edge, get_connector_points
from components.bridges.models import Connection
from components.coordinates.models import ConnectionType
from components.input.models import InputParameters
from utils.models import Vector, Ellipsoid


def _get_curve(params: InputParameters, point_from: Vector, point_to: Vector, connection_type: ConnectionType,
               n: int = 100) -> Tuple[Draft.BezCurve, Connection]:
    e: Ellipsoid = params.ellipsoid

    curve_direction: Vector = (point_to - point_from) / n
    point1 = get_point_on_socket_edge(params, point_from, curve_direction)
    point2 = get_point_on_socket_edge(params, point_to, -curve_direction)

    curve_direction = (point2 - point1) / n
    points = [point1 + i * curve_direction for i in range(n + 1)]
    points_on_ellipsoid = [e.closest_point_on_ellipsoid(p) for p in points]

    curve: Draft.BezCurve = Draft.make_bezcurve([App.Vector(x, y, z) for x, y, z in points_on_ellipsoid])
    connection = get_connector_points(params, point1, point2, connection_type)
    return curve, connection


def get_curves(params: InputParameters, points_coordinates: Dict[str, Vector],
               connected_points: List[Tuple[str,str,ConnectionType]]) -> Tuple[List[Draft.BezCurve], List[Connection]]:

    curves: List[Draft.BezCurve] = []
    connectors: List[Connection] = []

    for point_name_from, point_name_to, connection_type in connected_points:
        point_from = points_coordinates[point_name_from]
        point_to = points_coordinates[point_name_to]
        curve, connector = _get_curve(params, point_from, point_to, connection_type)
        curves.append(curve)
        connectors.append(connector)

    return curves, connectors