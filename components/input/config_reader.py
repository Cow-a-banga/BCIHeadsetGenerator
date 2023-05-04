import configparser
import math

from scipy.optimize import fsolve

from components.input.models import InputParameters, SocketModelParameters, BridgeModelParameters
from utils.models import Ellipsoid


def get_parameters_from_config(config_path: str) -> InputParameters:
    config = configparser.ConfigParser()
    config.read(config_path)

    bottom_length_param = config.getfloat('DEFAULT', 'BottomLength')
    longitudinal_length_param = config.getfloat('DEFAULT', 'LongitudinalLength')
    transverse_length_param = config.getfloat('DEFAULT', 'TransverseLength')
    socket_model_url = config.get('DEFAULT', 'ModelUrl', fallback="https://www.dropbox.com/s/4lbdo8gjrov5adk/NewCz.stl?dl=1")
    socket_model_path = config.get('DEFAULT', 'ModelPath', fallback="__TEST__.stl")
    socket_radius = config.getfloat('DEFAULT', 'SocketRadius', fallback=11)
    distance_to_connector = config.getfloat('DEFAULT', 'DistanceToConnector', fallback=12)
    bridge_width = config.getfloat('DEFAULT', 'BridgeWidth', fallback=4)
    bridge_height = config.getfloat('DEFAULT', 'BridgeHeght', fallback=4.5)
    export_folder_path = config.get('DEFAULT', 'ExportFolderPath',)

    def equations(p):
        x, y, z = p
        return (
            4 * (math.pi * x * y + (x - y) ** 2) / (x + y) - bottom_length_param,
            4 * (math.pi * z * y + (z - y) ** 2) / (z + y) - longitudinal_length_param,
            4 * (math.pi * x * z + (x - z) ** 2) / (x + z) - transverse_length_param
        )

    r1, r2, r3 = fsolve(equations, (1, 1, 1))

    ellipsoid = Ellipsoid(r1, r2, r3)
    socket = SocketModelParameters(url=socket_model_url, path=socket_model_path, radius=socket_radius)
    bridge = BridgeModelParameters(distance_to_connector, width=bridge_width, height=bridge_height)
    return InputParameters(ellipsoid, socket, bridge, export_folder_path)
