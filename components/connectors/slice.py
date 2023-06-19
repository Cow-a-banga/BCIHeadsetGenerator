from typing import List, Tuple
import Part
import FreeCAD as App
import numpy as np
from scipy.spatial.transform import Rotation as R
import BOPTools.SplitFeatures
import CompoundTools.Explode

from components.bridges.models import Connection, ConnectorPoint
from components.connectors.dovetail import generate_dovetail
from components.input.models import InputParameters


def _place_dovetails(params: InputParameters, connector_points: List[ConnectorPoint]) -> List[Part.Feature]:
    dovetails = []
    for connector_point in connector_points:
        point = connector_point.point
        vector = connector_point.direction
        dovetail = generate_dovetail(params.bridge.height, params.bridge.width, params.connector.length)
        normal = params.ellipsoid.get_normal(point)

        cross_product1 = normal.cross(vector.normalized()).normalized()
        cross_product2 = normal.cross(cross_product1).normalized()
        matrix = np.column_stack((cross_product1, cross_product2, normal))

        quaternions = R.from_matrix(matrix).as_quat()
        dovetail.Placement = App.Placement(App.Vector(point), App.Rotation(*quaternions))
        shape_obj = Part.show(dovetail)
        dovetails.append(shape_obj)
    return dovetails


def _slice_bridge(bridge_obj: Part.Feature, dovetails: List[Part.Feature]) -> None:
    f = BOPTools.SplitFeatures.makeSlice(name='Slice')
    f.Base = bridge_obj
    f.Tools = dovetails
    f.Mode = 'Split'
    f.Proxy.execute(f)
    f.purgeTouched()
    for obj in f.ViewObject.Proxy.claimChildren():
        obj.ViewObject.hide()
    CompoundTools.Explode.explodeCompound(f)
    f.ViewObject.hide()


def _generate_slice_shape_names(shape_num: int, dovetails_num: int) -> Tuple[str, List[str]]:
    slice_num_text = f"{shape_num:03d}"
    slice_num_text = slice_num_text if slice_num_text != "000" else ""

    shape_name = f"Shape{slice_num_text}"
    slice_name = [f"Slice{slice_num_text}_child{i}" for i in range(dovetails_num + 1)]
    return shape_name, slice_name


def slice_bridges(params: InputParameters, bridges: List[Part.Shape], connections: List[Connection],
                  doc: App.Document) -> List[List[str]]:
    slice_names = []
    shape_names = []
    shape_num = 0

    for index, connection in enumerate(connections):
        bridge_shape = Part.show(bridges[index])
        dovetails = _place_dovetails(params, connection.points)

        _slice_bridge(bridge_shape, dovetails)

        shape_name, slice_name = _generate_slice_shape_names(shape_num, len(dovetails))
        shape_names.append(shape_name)
        slice_names.append(slice_name)
        shape_num += 1

    for name in shape_names:
        doc.removeObject(name)
    return slice_names

__all__ = ["slice_bridges"]
