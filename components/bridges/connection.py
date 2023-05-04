from typing import List

from components.bridges.models import Connection, ConnectorPoint
from components.coordinates.models import ConnectionType
from components.input.models import InputParameters
from utils.models import Vector, Ellipsoid


def get_point_on_socket_edge(params: InputParameters, socket_center: Vector, curve_direction: Vector) -> Vector:
    socket_normal = params.ellipsoid.get_normal(socket_center)
    socket_center_to_edge:Vector = curve_direction - (curve_direction.dot(socket_normal)) * socket_normal
    socket_center_to_edge = socket_center_to_edge.normalized() * params.socket.radius
    socket_edge_point = socket_center + socket_center_to_edge
    return params.ellipsoid.closest_point_on_ellipsoid(socket_edge_point)


def get_connector_points(params: InputParameters, point_from: Vector, point_to: Vector,
                         connection_type: ConnectionType) -> Connection:
    e:Ellipsoid = params.ellipsoid
    points: List[ConnectorPoint] = []
    if connection_type == ConnectionType.OneCut:
        center:Vector = (point_from * 0.6 + point_to * 0.4)
        center = e.closest_point_on_ellipsoid(center)
        points = [ConnectorPoint(center, center-point_from)]
    elif connection_type == ConnectionType.TwoCuts:
        connector_point_vector = (point_to - point_from).normalized() * params.bridge.distance_to_connector
        p1 = e.closest_point_on_ellipsoid(point_from + connector_point_vector)
        p2 = e.closest_point_on_ellipsoid(point_to - connector_point_vector)
        points = [ConnectorPoint(p1, p1 - point_from), ConnectorPoint(p2, p2 - point_to)]
    return Connection(connection_type, points)