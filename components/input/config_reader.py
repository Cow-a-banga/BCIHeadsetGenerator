import configparser
import math

from scipy.optimize import fsolve

from components.input.models import InputParameters, SocketModelParameters, BridgeModelParameters
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
    text_offset = config.getfloat('Model', 'TextOffset', fallback=1)
    text_width = config.getfloat('Model', 'TextWidth', fallback=0.5)
    export_folder_path = config.get('Model', 'ExportFolderPath', fallback="_export")

    lower_head_circumference = config.getfloat('InputFromCircumference', 'LowerHeadCircumference')
    anterior_posterior_circumference = config.getfloat('InputFromCircumference', 'AnteriorPosteriorCircumference')
    left_right_circumference = config.getfloat('InputFromCircumference', 'LeftRightCircumference')

    up_down_radius = config.getfloat('InputFromRadius', 'UpDownRadius')
    left_right_radius = config.getfloat('InputFromRadius', 'LeftRightRadius')
    near_far_radius = config.getfloat('InputFromRadius', 'NearFarRadius')

    def equations(p):
        x, y, z = p
        return (
            4 * (math.pi * x * y + (x - y) ** 2) / (x + y) - lower_head_circumference,
            4 * (math.pi * z * y + (z - y) ** 2) / (z + y) - anterior_posterior_circumference,
            4 * (math.pi * x * z + (x - z) ** 2) / (x + z) - left_right_circumference
        )

    if up_down_radius is None and left_right_radius is None and near_far_radius is None:
        r1, r2, r3 = fsolve(equations, (1, 1, 1))

        ellipsoid = Ellipsoid(r1, r2, r3)
    else:
        ellipsoid = Ellipsoid(left_right_radius, near_far_radius, up_down_radius)

    socket = SocketModelParameters(url=socket_model_url, path=socket_model_path, radius=socket_radius)
    bridge = BridgeModelParameters(distance_to_connector, width=bridge_width, height=bridge_height, text_offset=text_offset, text_width=text_width)
    return InputParameters(ellipsoid, socket, bridge, export_folder_path)
