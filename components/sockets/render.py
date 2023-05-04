import os
import urllib.request
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation
import Mesh
import FreeCAD as App

from components.input.models import InputParameters
from utils.functions import rotation_matrix_from_vectors
from utils.models import Vector, Ellipsoid


def _get_rotation_quaternion(e: Ellipsoid, coords: Vector) -> np.ndarray:
    up = Vector(0, 0, 1)
    normal = e.get_normal(coords)
    matrix = rotation_matrix_from_vectors(up, normal)
    return Rotation.from_matrix(matrix).as_quat()


def _render_socket(name: str, coords: Vector, params: InputParameters,) -> Mesh.Mesh:
    socket_mesh = Mesh.Mesh(params.socket.path)
    quat = _get_rotation_quaternion(params.ellipsoid, coords)

    rotation = App.Rotation(quat[0], quat[1], quat[2], quat[3])
    vector = App.Vector(coords[0], coords[1], coords[2])
    socket_mesh.Placement = App.Placement(vector, rotation)
    Mesh.show(socket_mesh, name)
    return socket_mesh


def render_sockets(params: InputParameters, points_coordinates: Dict[str, Vector]) -> List[Mesh.Mesh]:
    with urllib.request.urlopen(params.socket.url) as response, open(params.socket.path, 'wb') as file:
        file.write(response.read())
        points_mesh = [_render_socket(name, coords, params) for name, coords in points_coordinates.items()]
    os.remove(params.socket.path)
    return points_mesh


__all__ = ["render_sockets"]
