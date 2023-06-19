from utils.models import Ellipsoid
from dataclasses import dataclass


@dataclass
class BridgeModelParameters:
    distance_to_connector: float
    width: float
    height: float
    text_offset: float
    text_width: float


@dataclass
class ConnectorParameters:
    length: float
    max_width: float
    min_width: float


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
    connector: ConnectorParameters
    export_folder_path: str
