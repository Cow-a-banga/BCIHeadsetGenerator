import itertools
import os
from typing import List, Dict
from components.coordinates.models import get_connected_points

import Mesh
import FreeCAD as App

from utils.models import Vector


def _export_bridge(distances: List[float], objects, coords, socket_names: List[str], text, export_folder_path: str) -> None:
    max_dist = max(distances)
    if distances[0] == max_dist:
        bridge = objects[2]
        objects.pop(2)
        coords.pop(2)
    elif distances[1] == max_dist:
        bridge = objects[1]
        objects.pop(1)
        coords.pop(1)
    elif distances[2] == max_dist:
        bridge = objects[0]
        objects.pop(0)
        coords.pop(0)

    Mesh.export([bridge, *text], os.path.join(export_folder_path, f"{socket_names[0]}_{socket_names[1]}.stl"))


def _add_used_sockets(from_coord: Vector, to_coord: Vector, coords, objects, socket_names, texts, used_sockets, doc) -> None:
    if from_coord.distance(coords[0]) < to_coord.distance(coords[0]):
        objects_for_sockets = objects
    else:
        objects_for_sockets = list(reversed(objects))

    for i, (name, text) in enumerate(zip(socket_names, texts)):
        if name in used_sockets:
            used_sockets[name].append(objects_for_sockets[i])
        else:
            socket = doc.getObject(name)
            used_sockets[name] = [socket, objects_for_sockets[i]]
        used_sockets[name].append(text)


def _export_sockets(used_sockets, export_folder_path):
    for name in used_sockets:
        Mesh.export(used_sockets[name], os.path.join(export_folder_path, f"{name}.stl"))


def _prepare_parts(points_coordinates, points, names, text, used_sockets, export_path:str, doc):
    socket_names = [points[0], points[1]]
    objects = [doc.getObject(name) for name in names]
    coords = [obj.Shape.CenterOfMass for obj in objects]
    coords = [Vector(coord.x, coord.y, coord.z) for coord in coords]
    from_coord = points_coordinates[points[0]]
    to_coord = points_coordinates[points[1]]
    pairs = list(itertools.combinations(coords, 2))
    distances = [p1.distance(p2) for (p1, p2) in pairs]

    if len(names) == 3:
        _export_bridge(distances, objects, coords, socket_names, [text[0][0], text[1][1]], export_path)

    socket_text = [text[0][1], text[1][0]] if len(text) == 2 else [text[0][1], text[0][0]]
    _add_used_sockets(from_coord, to_coord, coords, objects, socket_names, socket_text, used_sockets, doc)


def export(slice_names: List[List[str]], points_coordinates: Dict[str, Vector], texts, export_path:str, doc: App.Document) -> None:
    used_sockets = {}
    for names, points, text in zip(slice_names, get_connected_points(), texts):
        _prepare_parts(points_coordinates, points, names, text, used_sockets, export_path, doc)
    _export_sockets(used_sockets, export_path)
    doc.saveAs(os.path.join(export_path, "_eeg_headset.FCStd"))


__all__ = ["export"]

