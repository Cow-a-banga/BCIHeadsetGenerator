import configparser
import math

from scipy.optimize import fsolve

from components.input.models import InputParameters, SocketModelParameters, BridgeModelParameters, ConnectorParameters
from utils.models import Ellipsoid


def get_parameters_from_config(config_path: str) -> InputParameters:
    config = configparser.ConfigParser()
    config.read(config_path)

    socket_model_url = config.get('Model', 'ModelUrl', fallback="https://www.dropbox.com/s/4lbdo8gjrov5adk/NewCz.stl?dl=1")
    socket_model_path = config.get('Model', 'ModelPath', fallback="__TEST__.stl")
    socket_radius = config.getfloat('Model', 'SocketRadius', fallback=11)
    distance_to_connector = config.getfloat('Model', 'DistanceToConnector', fallback=12)
    bridge_width = config.getfloat('Model', 'BridgeWidth', fallback=4)
    bridge_height = config.getfloat('Model', 'BridgeHeight', fallback=4.5)
    connector_length = config.getfloat('Model', 'ConnectorLength', fallback=6.25)
    text_offset = config.getfloat('Model', 'TextOffset', fallback=1)
    text_width = config.getfloat('Model', 'TextWidth', fallback=0.5)
    export_folder_path = config.get('Model', 'ExportFolderPath', fallback="_export")

    lower_head_circumference = config.getfloat('InputFromCircumference', 'LowerHeadCircumference', fallback=None)
    anterior_posterior_circumference = config.getfloat('InputFromCircumference', 'AnteriorPosteriorCircumference', fallback=None)
    left_right_circumference = config.getfloat('InputFromCircumference', 'LeftRightCircumference', fallback=None)

    up_down_radius = config.getfloat('InputFromRadius', 'UpDownRadius', fallback=None)
    left_right_radius = config.getfloat('InputFromRadius', 'LeftRightRadius', fallback=None)
    near_far_radius = config.getfloat('InputFromRadius', 'NearFarRadius', fallback=None)

    def equations(p):
        x, y, z = p
        return (
            math.pi * (3 * (x + y) - math.sqrt((3*x + y)*(3*y + x))) - lower_head_circumference,
            math.pi * (3 * (z + y) - math.sqrt((3*z + y)*(3*y + z))) - anterior_posterior_circumference,
            math.pi * (3 * (x + z) - math.sqrt((3*x + z)*(3*z + x))) - left_right_circumference
        )

    if up_down_radius is None and left_right_radius is None and near_far_radius is None:
        r1, r2, r3 = fsolve(equations, (1, 1, 1))
        print(r1, r2, r3)

        ellipsoid = Ellipsoid(r1, r2, r3)
    else:
        ellipsoid = Ellipsoid(left_right_radius, near_far_radius, up_down_radius)

    socket = SocketModelParameters(url=socket_model_url, path=socket_model_path, radius=socket_radius)
    bridge = BridgeModelParameters(distance_to_connector, width=bridge_width, height=bridge_height, text_offset=text_offset, text_width=text_width)
    connector = ConnectorParameters(length=connector_length, max_width=10, min_width=10)
    return InputParameters(ellipsoid, socket, bridge, connector, export_folder_path)
