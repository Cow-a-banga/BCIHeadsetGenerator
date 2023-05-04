from utils.models import Ellipsoid
from dataclasses import dataclass


@dataclass
class BridgeModelParameters:
    distance_to_connector: float
    width: float
    height: float


@dataclass
class SocketModelParameters:
    url: str
    path: str
    radius: float


@dataclass
class InputParameters:
    ellipsoid: Ellipsoid
    socket: SocketModelParameters
    bridge: BridgeModelParameters
    export_folder_path: str
