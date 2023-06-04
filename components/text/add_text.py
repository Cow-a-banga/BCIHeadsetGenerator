from typing import List

import FreeCAD as App
import Draft
import numpy as np
from scipy.spatial.transform import Rotation as R

from components.bridges.models import Connection, ConnectorPoint
from components.coordinates.models import ConnectionType
from components.input.models import InputParameters


def _get_position(params: InputParameters, connection: ConnectorPoint, directionReverse: int):
    point = connection.point
    vector = directionReverse * connection.direction.normalized()
    normal = params.ellipsoid.get_normal(point)
    new_z = normal.cross(vector).normalized()
    new_x = normal.cross(new_z).normalized()
    matrix = np.column_stack((new_x, normal, new_z))

    quaternions = R.from_matrix(matrix).as_quat()
    position = connection.point + (params.bridge.width / 2 * new_z) + (params.bridge.height / 4 * normal)

    return quaternions, position, params.bridge.text_width * new_z


def _generate_text(params: InputParameters, text_str1: str, text_str2: str, connection: ConnectorPoint,
                   directionReverse: int):
    text1 = Draft.make_shapestring(String=text_str1, FontFile="C:/Windows/Fonts/Arial.ttf", Size=2.0, Tracking=0.0)
    text2 = Draft.make_shapestring(String=text_str2, FontFile="C:/Windows/Fonts/Arial.ttf", Size=2.0, Tracking=0.0)

    width = text2.Shape.BoundBox.XLength

    quat, position, new_z = _get_position(params, connection, directionReverse)
    if directionReverse == -1:
        text1.Placement = App.Placement(App.Vector(position + params.bridge.text_offset * connection.direction), App.Rotation(*quat))
        text2.Placement = App.Placement(App.Vector(position - (width + params.bridge.text_offset) * connection.direction),
                                        App.Rotation(*quat))
    else:
        text1.Placement = App.Placement(App.Vector(position - params.bridge.text_offset * connection.direction), App.Rotation(*quat))
        text2.Placement = App.Placement(App.Vector(position + ((width + params.bridge.text_offset) * connection.direction)),
                                        App.Rotation(*quat))

    extruded_text1 = Draft.extrude(text1, App.Vector(*new_z))
    extruded_text2 = Draft.extrude(text2, App.Vector(*new_z))
    return [extruded_text1, extruded_text2]


def add_text(params: InputParameters, connections: List[Connection]):
    texts = []
    for connection in connections:
        texts_in_bridge = []
        if connection.type == ConnectionType.TwoCuts:
            texts_in_bridge.append(_generate_text(params, connection.from_name, connection.to_name, connection.points[0], -1))
            texts_in_bridge.append(_generate_text(params, connection.from_name, connection.to_name, connection.points[1], 1))
        elif connection.type == ConnectionType.OneCut:
            texts_in_bridge.append(_generate_text(params, connection.from_name, connection.to_name, connection.points[0], -1))
        texts.append(texts_in_bridge)
    return texts





