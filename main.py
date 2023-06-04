import os
import FreeCAD as App

from components.bridges.render import render_bridges
from components.connectors.slice import slice_bridges
from components.coordinates.models import get_coordinates
from components.export.export import export
from components.input.config_reader import get_parameters_from_config
from components.input.models import InputParameters
from components.sockets.render import render_sockets
from components.text.add_text import add_text

def generate_bci_headset(input_parameters: InputParameters):
    doc = App.newDocument()
    points_coordinates = get_coordinates(input_parameters.ellipsoid)
    render_sockets(input_parameters, points_coordinates)
    bridges, connector_points = render_bridges(input_parameters, points_coordinates, doc)
    slice_names = slice_bridges(input_parameters, bridges, connector_points, doc)
    texts = add_text(input_parameters, connector_points)
    doc.recompute()
    export(slice_names, points_coordinates, texts, input_parameters.export_folder_path, doc)


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
    input_parameters = get_parameters_from_config(config_path)
    generate_bci_headset(input_parameters)


if __name__ == '__main__':
    main()
