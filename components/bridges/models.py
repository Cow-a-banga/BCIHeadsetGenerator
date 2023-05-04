from dataclasses import dataclass
from typing import List

from components.coordinates.models import ConnectionType
from utils.models import Vector


@dataclass
class ConnectorPoint:
    point: Vector
    direction: Vector


@dataclass
class Connection:
    type: ConnectionType
    points: List[ConnectorPoint]