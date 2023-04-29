import FreeCAD as App
import Part
import Mesh

width = 4
height = 5
w = width/2

length = w/2
out_length = 2
out_height = 2

def get_points(z_level):
    return [
        App.Vector(-(w + out_length), 0, z_level),
        App.Vector(-0.25 * w, 0, z_level),
        App.Vector(-0.75 * w, length, z_level),
        App.Vector(0.75 * w, length, z_level),
        App.Vector(0.25 * w, 0, z_level),
        App.Vector((w + out_length), 0, z_level),
    ]

wire1 = Part.makePolygon(get_points(-out_height))
wire2 = Part.makePolygon(get_points(height + out_height))

surface = Part.makeRuledSurface(wire1, wire2)

Part.show(surface)

App.ActiveDocument.recompute()

mesh = Mesh.Mesh(surface.tessellate(1.0))
mesh.write("C:\\Users\\chiru\\AppData\\Roaming\\FreeCAD\\Macro\\template.stl")
#Part.export([surface], )