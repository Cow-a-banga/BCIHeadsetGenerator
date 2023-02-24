import FreeCAD as App
import math
import Mesh
from scipy.spatial.transform import Rotation as R
import numpy as np
import Draft
import Part
import configparser
import os
from scipy.optimize import fsolve
import urllib.request

def normalize(vec):
    l = (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)**0.5
    return [vec[0]/l, vec[1]/l, vec[2]/l]
  
def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3)

def ellipsoidFormula(r1, r2, r3, x, y):
    formula = 1 - (x/r1)**2 - (y/r2)**2
    if formula< 0:
        return [x, y, 0]
    return [x, y, r3 * math.sqrt(formula)]

def ellipsFormula(r1, r2, x, sign):
    return sign * r2 * math.sqrt(1 - (x/r1) ** 2)

def getCoordinates(r1, r2, r3):
    return { 
    "Cz":[0, 0, r3],
    "T3":[r1, 0, 0],
    "T4":[-r1, 0, 0],
    "C3":[r1*math.cos(math.pi/4), 0, r3*math.sin(math.pi/4)],
    "C4":[-r1*math.cos(math.pi/4), 0, r3*math.sin(math.pi/4)],
    "Pz":[0, r2*math.cos(math.pi/4), r3*math.sin(math.pi/4)],
    "Fz":[0, -r2*math.cos(math.pi/4), r3*math.sin(math.pi/4)],
    "O1":[-r1*math.cos(-3*math.pi/5), -r2*math.sin(-3*math.pi/5), 0],
    "O2":[-r1*math.cos(-2*math.pi/5), -r2*math.sin(-2*math.pi/5), 0],
    "Fp1":[-r1*math.cos(3*math.pi/5), -r2*math.sin(3*math.pi/5), 0],
    "Fp2":[-r1*math.cos(2*math.pi/5), -r2*math.sin(2*math.pi/5), 0],
    "T5":[-r1*math.cos(-4*math.pi/5), -r2*math.sin(-4*math.pi/5), 0],
    "T6":[-r1*math.cos(-math.pi/5), -r2*math.sin(-math.pi/5), 0],
    "F7":[-r1*math.cos(4*math.pi/5), -r2*math.sin(4*math.pi/5), 0],
    "F8":[-r1*math.cos(math.pi/5), -r2*math.sin(math.pi/5), 0],
    "P3":ellipsoidFormula(r1, r2, r3, -r1*math.cos(3*math.pi/5), -r1*math.cos(4*math.pi/5)),
    "F3":ellipsoidFormula(r1, r2, r3, -r1*math.cos(3*math.pi/5), r1*math.cos(4*math.pi/5)),
    "P4":ellipsoidFormula(r1, r2, r3, r1*math.cos(3*math.pi/5), -r1*math.cos(4*math.pi/5)),
    "F4":ellipsoidFormula(r1, r2, r3, r1*math.cos(3*math.pi/5), r1*math.cos(4*math.pi/5)),
    }
    
def getConnectedPoints():
    return [
        [
            ("Fp1", "Fp2"),
            ("Fp1", "F7"),
            ("F7", "T3"),
            ("T3", "T5"),
            ("T5", "O1"),
            ("Fp2", "F8"),
            ("F8", "T4"),
            ("T4", "T6"),
            ("T6", "O2"),
            ("O1", "O2")
        ],
        [
            ("Fp1", "F3"),
            ("F7", "F3"),
            ("Fz", "F3"),
            ("C3", "F3"),
            ("T3", "C3"),
            ("Cz", "C3"),
            ("P3", "C3"),
            ("T5", "P3"),
            ("Pz", "P3"),
            ("O1", "P3"),
            ("Fp2", "F4"),
            ("F8", "F4"),
            ("Fz", "F4"),
            ("C4", "F4"),
            ("T4", "C4"),
            ("Cz", "C4"),
            ("P4", "C4"),
            ("T6", "P4"),
            ("Pz", "P4"),
            ("O2", "P4"),
            ("Fz", "Cz"),
            ("Pz", "Cz")
        ]
    ]

def getInterPointsX(p1, p2, dx):
    inversed = False
    if p1 > p2:
        p1, p2 = p2, p1
        inversed = True
    points = []
    i = p1
    while i + dx < p2:
        points.append(i)
        i += dx
    points.append(p2)
    if inversed:
        points.reverse()
    return points

def getInterPoints(p1, p2, points_coordinates, dx):
    coord1 = points_coordinates[p1]
    coord2 = points_coordinates[p2]
    inter_points_x = getInterPointsX(coord1[0], coord2[0], dx)
    inter_points = [App.Vector(x, ellipsFormula(r2, r3, x, math.copysign(1, coord1[1])), 0) for x in inter_points_x]
    return inter_points


def getCurvesOnTheBottom(r2, r3, points_coordinates, dx = 0.5):
    connections = getConnectedPoints()
    curves = []
    for p1, p2 in connections[0]:
        coord1 = points_coordinates[p1]
        coord2 = points_coordinates[p2]
        inter_points_x = getInterPointsX(coord1[0], coord2[0], dx)

        inter_points = [App.Vector(x, ellipsFormula(r2, r3, x, math.copysign(1, coord1[1])), 0) for x in inter_points_x]

        curve = Draft.make_bezcurve(inter_points)
        curves.append(curve.Shape)
    return curves

def getABForLine(p1, p2):
    if abs(p1[0] - p2[0]) < 1e-10 :
        return [p1[0], None]

    a = (p2[1] - p1[1])/(p2[0] - p1[0])
    b = p1[1] - a*p1[0]
    return [a,b]

def getCurvesNotOnTheBottom(r1, r2, r3, points_coordinates, dx = 0.5):
    connections = getConnectedPoints()
    curves = []
    for p1, p2 in connections[1]:
        print(p1,p2)

        coord1 = points_coordinates[p1]
        coord2 = points_coordinates[p2]

        [a,b] = getABForLine(coord1, coord2)

        if b != None:
            inter_points_x = getInterPointsX(coord1[0], coord2[0], dx)
            inter_points_xy = [[x, a*x + b] for x in inter_points_x]
        else:
            inter_points_y = getInterPointsX(coord1[1], coord2[1], dx)
            inter_points_xy = [[a, y] for y in inter_points_y]

        inter_points_xyz = [ellipsoidFormula(r1,r2,r3, xy[0], xy[1]) for xy in inter_points_xy]

        inter_points = [App.Vector(xyz[0], xyz[1], xyz[2]) for xyz in inter_points_xyz]

        curve = Draft.make_bezcurve(inter_points)
        curves.append(curve.Shape)
    return curves

def renderSocktes(r1, r2, r3, points_coordinates):
    points_mesh = []
    urllib.request.urlretrieve(config['DEFAULT']['ModelUrl'], config['DEFAULT']['ModelPath'])
    for _,coords in points_coordinates.items():
        point = Mesh.Mesh(config['DEFAULT']['ModelPath'])
        
        normal = [2*coords[0]/r1**2, 2*coords[1]/r2**2, 2*coords[2]/r3**2]
        normalized_normal = normalize(normal)
        matrix = rotation_matrix_from_vectors(np.array([0,0,1]), np.array(normalized_normal))
        angles = R.from_matrix(matrix).as_euler('zyx', degrees=True)
        point.Placement = App.Placement(App.Vector(coords[0], coords[1], coords[2]),  App.Rotation(-angles[0], angles[1], angles[2]))
        points_mesh.append(point)
        Mesh.show(point)
        
        #draw normals
        # p1 = App.Vector(coords[0], coords[1], coords[2])
        # p2 = App.Vector(coords[0] + 100 * normalized_normal[0], coords[1] + 100 * normalized_normal[1], coords[2] + 100 * normalized_normal[2])
        # Draft.make_line(p1, p2)
    os.remove(config["DEFAULT"]["ModelPath"])
    return points_mesh

def renderBridges(r1, r2, r3, points_coordinates):
    curves0 = getCurvesOnTheBottom(r1, r2, points_coordinates)
    curves1 = getCurvesNotOnTheBottom(r1,r2,r3,points_coordinates, dx = 1)

    dr = 10
    r1+=dr
    r2+=dr
    r3+=dr

    points_coordinates2 = getCoordinates(r1, r2, r3)
    curves02 = getCurvesOnTheBottom(r1, r2, points_coordinates2)
    curves12 = getCurvesNotOnTheBottom(r1,r2,r3,points_coordinates2, dx = 1)

    surfaces0 = []
    surfaces1 = []

    for i in range(len(curves0)):
        surface = Part.makeRuledSurface(curves0[i], curves02[i])
        e = surface.extrude(App.Vector(0,0,5))
        Part.show(e)
        surfaces0.append(e)

    for i in range(len(curves1)):
        surface = Part.makeRuledSurface(curves1[i], curves12[i])
        Part.show(surface)
        surfaces1.append(surface)
    return surfaces0, surfaces1


def equations(p):
    x, y, z = p
    return (
        4*(math.pi*x*y + (x-y)**2)/(x+y) - BottomLengthParam,
        4*(math.pi*z*y + (z-y)**2)/(z+y) - LongitudinalLengthParam, 
        4*(math.pi*x*z + (x-z)**2)/(x+z) - TransverseLengthParam
        )


config = configparser.ConfigParser()
#config.read(os.environ["BCICONFIGPATH"])
config.read('C:/Users/chiru/AppData/Roaming/FreeCAD/Macro/config.ini')
BottomLengthParam = float(config['DEFAULT']['BottomLength'])
LongitudinalLengthParam = float(config['DEFAULT']['LongitudinalLength'])
TransverseLengthParam = float(config['DEFAULT']['TransverseLength'])


r1, r2, r3 =  fsolve(equations, (1, 1, 1))

doc = App.activeDocument()
ellipsoid = doc.addObject("Part::Ellipsoid", "myEllipsoid")
ellipsoid.Radius1 = r3
ellipsoid.Radius2 = r1
ellipsoid.Radius3 = r2

dr=2
r1+=dr
r2+=dr
r3+=dr

points_coordinates = getCoordinates(r1, r2, r3)
renderBridges(r1,r2,r3, points_coordinates)
renderSocktes(r1,r2,r3, points_coordinates)

doc.recompute()