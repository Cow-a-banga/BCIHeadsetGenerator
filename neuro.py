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
from scipy.optimize import newton
from enum import Enum
import BOPTools.SplitAPI
import BOPTools.SplitFeatures
import CompoundTools.Explode
import itertools

class ConnectionType(Enum):
    OneCut = 1
    TwoCuts = 2

def getNormal(r1,r2,r3, coords):
    return np.array([2*coords[0]/r1**2, 2*coords[1]/r2**2, 2*coords[2]/r3**2])

def projectionOnEllipsoid(r1, r2, r3, coords):
    return coords * np.sqrt(1 / (r1**2 * coords[0]**2 + r2**2 * coords[1]**2 + r3**2 * coords[2]**2))

def normalize(vec):
    return vec / np.linalg.norm(vec)
  
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
    if formula < 0:
        return np.array([x, y, 0])
    return np.array([x, y, r3 * math.sqrt(formula)])

def ellipsFormula(r1, r2, x, sign):
    return sign * r2 * math.sqrt(1 - (x/r1) ** 2)

def getCoordinates(r1, r2, r3):
    return { 
    "Cz":np.array([0, 0, r3]),
    "T3":np.array([r1, 0, 0]),
    "T4":np.array([-r1, 0, 0]),
    "C3":np.array([r1*math.cos(math.pi/4), 0, r3*math.sin(math.pi/4)]),
    "C4":np.array([-r1*math.cos(math.pi/4), 0, r3*math.sin(math.pi/4)]),
    "Pz":np.array([0, r2*math.cos(math.pi/4), r3*math.sin(math.pi/4)]),
    "Fz":np.array([0, -r2*math.cos(math.pi/4), r3*math.sin(math.pi/4)]),
    "O1":np.array([-r1*math.cos(-3*math.pi/5), -r2*math.sin(-3*math.pi/5), 0]),
    "O2":np.array([-r1*math.cos(-2*math.pi/5), -r2*math.sin(-2*math.pi/5), 0]),
    "Fp1":np.array([-r1*math.cos(3*math.pi/5), -r2*math.sin(3*math.pi/5), 0]),
    "Fp2":np.array([-r1*math.cos(2*math.pi/5), -r2*math.sin(2*math.pi/5), 0]),
    "T5":np.array([-r1*math.cos(-4*math.pi/5), -r2*math.sin(-4*math.pi/5), 0]),
    "T6":np.array([-r1*math.cos(-math.pi/5), -r2*math.sin(-math.pi/5), 0]),
    "F7":np.array([-r1*math.cos(4*math.pi/5), -r2*math.sin(4*math.pi/5), 0]),
    "F8":np.array([-r1*math.cos(math.pi/5), -r2*math.sin(math.pi/5), 0]),
    "P3":np.array(ellipsoidFormula(r1, r2, r3, -r1*math.cos(3*math.pi/5), -r1*math.cos(4*math.pi/5))),
    "F3":np.array(ellipsoidFormula(r1, r2, r3, -r1*math.cos(3*math.pi/5), r1*math.cos(4*math.pi/5))),
    "P4":np.array(ellipsoidFormula(r1, r2, r3, r1*math.cos(3*math.pi/5), -r1*math.cos(4*math.pi/5))),
    "F4":np.array(ellipsoidFormula(r1, r2, r3, r1*math.cos(3*math.pi/5), r1*math.cos(4*math.pi/5))),
    }
      
def getConnectedPoints():
    return [
            ("Fp1", "Fp2", ConnectionType.TwoCuts),
            ("Fp1", "F7", ConnectionType.TwoCuts),
            ("F7", "T3", ConnectionType.TwoCuts),
            ("T3", "T5", ConnectionType.TwoCuts),
            ("T5", "O1", ConnectionType.TwoCuts),
            ("Fp2", "F8", ConnectionType.TwoCuts),
            ("F8", "T4", ConnectionType.TwoCuts),
            ("T4", "T6", ConnectionType.TwoCuts),
            ("T6", "O2", ConnectionType.TwoCuts),
            ("O1", "O2", ConnectionType.TwoCuts),
            ("Fp1", "F3", ConnectionType.TwoCuts),
            ("F7", "F3", ConnectionType.TwoCuts),
            ("Fz", "F3", ConnectionType.OneCut),
            ("C3", "F3", ConnectionType.TwoCuts),
            ("T3", "C3", ConnectionType.TwoCuts),
            ("Cz", "C3", ConnectionType.TwoCuts),
            ("P3", "C3", ConnectionType.TwoCuts),
            ("T5", "P3", ConnectionType.TwoCuts),
            ("Pz", "P3", ConnectionType.OneCut),
            ("O1", "P3", ConnectionType.TwoCuts),
            ("Fp2", "F4", ConnectionType.TwoCuts),
            ("F8", "F4", ConnectionType.TwoCuts),
            ("Fz", "F4", ConnectionType.OneCut),
            ("C4", "F4", ConnectionType.TwoCuts),
            ("T4", "C4", ConnectionType.TwoCuts),
            ("Cz", "C4", ConnectionType.TwoCuts),
            ("P4", "C4", ConnectionType.TwoCuts),
            ("T6", "P4", ConnectionType.TwoCuts),
            ("Pz", "P4", ConnectionType.OneCut),
            ("O2", "P4", ConnectionType.TwoCuts),
            ("Fz", "Cz", ConnectionType.TwoCuts),
            ("Pz", "Cz", ConnectionType.TwoCuts)
    ]

def closest_point_on_ellipsoid(point, a, b, c):
    normal = normalize(getNormal(r1,r2,r3,point))

    def ellipsoidForCloset(x, y, z, a, b, c):
        return (x / a)**2 + (y / b)**2 + (z / c)**2 - 1
    
    def f(t):
        x = point[0] + t * normal[0]
        y = point[1] + t * normal[1]
        z = point[2] + t * normal[2]
        return ellipsoidForCloset(x, y, z, a, b, c)
    t = newton(f, 0)
    return point + t * normal


def getPointOnSocketEdge(point, vector, socketRadius, r1, r2, r3):
        n1 = normalize(getNormal(r1,r2,r3, point))
        vectorInPlane = vector-(vector.dot(n1))*n1
        vectorInPlane = normalize(vectorInPlane)*socketRadius
        point1 = point + vectorInPlane
        return closest_point_on_ellipsoid(point1, r1,r2,r3)

def addConnectorPointsToList(point1, point2, type, connectorPoints):
    if(type == ConnectionType.OneCut):
        center = (point1 * 0.6 + point2 * 0.4)
        center = closest_point_on_ellipsoid(center, r1,r2,r3)
        connectorPoints.append([(center, center - point1)])
    elif(type == ConnectionType.TwoCuts):
        DISTANCE_TILL_CONNECTOR = 12
        connector_point_vector = normalize(point2-point1) * DISTANCE_TILL_CONNECTOR
        p1 = closest_point_on_ellipsoid(point1 + connector_point_vector, r1,r2,r3)
        p2 = closest_point_on_ellipsoid(point2 - connector_point_vector, r1,r2,r3)
        connectorPoints.append([(p1,p1 - point1),(p2,p2-point2)])

def getCurves(r1, r2, r3, points_coordinates, connections, addConnectorPoints = False, n = 100):
    curves = []
    connectorPoints = []
    socketRadius = 11

    for p1, p2, type in connections:
        coord1 = points_coordinates[p1]
        coord2 = points_coordinates[p2]
        n_vector = (coord2-coord1)/n

        point1 = getPointOnSocketEdge(coord1, n_vector, socketRadius,r1,r2,r3)
        point2 = getPointOnSocketEdge(coord2, -n_vector, socketRadius,r1,r2,r3)
        
        points = [point1]
        n_vector = (point2 - point1)/n
        for i in range(1, n):
            points.append(point1 + i*n_vector)
        points.append(point2)

        if(addConnectorPoints):
            addConnectorPointsToList(point1, point2, type, connectorPoints)

        points_on_ellipsoid = [closest_point_on_ellipsoid(p, r1,r2,r3) for p in points]

        curve = Draft.make_bezcurve([App.Vector(x,y,z) for x,y,z in points_on_ellipsoid])
        curves.append(curve)

    return curves, connectorPoints

def renderSocktes(r1, r2, r3, points_coordinates):
    points_mesh = []
    urllib.request.urlretrieve(config['DEFAULT']['ModelUrl'], config['DEFAULT']['ModelPath'])
    for name,coords in points_coordinates.items():
        point = Mesh.Mesh(config['DEFAULT']['ModelPath'])

        up = np.array([0,0,1])
        normal = normalize(getNormal(r1,r2,r3, coords))

        matrix = rotation_matrix_from_vectors(up, normal)
        quat = R.from_matrix(matrix).as_quat()
        point.Placement = App.Placement(App.Vector(coords[0], coords[1], coords[2]),  App.Rotation(quat[0], quat[1], quat[2], quat[3]))
        points_mesh.append(point)
        Mesh.show(point, name)
    os.remove(config["DEFAULT"]["ModelPath"])
    return points_mesh

def renderBridges(r1, r2, r3, points_coordinates):
    connections = getConnectedPoints()
    curvesInner, connectorPoints = getCurves(r1,r2,r3, points_coordinates, connections, addConnectorPoints=True)

    models = []
    offsetSize = 2
    height = 4.5
    for curve, points in zip(curvesInner, connectorPoints):
        point, _ = points[0]
        normal = normalize(getNormal(r1,r2,r3,point)) * height
        feature = doc.addObject("Part::Feature", "Wire")
        feature.Shape = curve.Shape
        cloned_object = doc.copyObject(feature)
        cloned_object.Placement.Base.x += normal[0]
        cloned_object.Placement.Base.y += normal[1]
        cloned_object.Placement.Base.z += normal[2]

        surface = Part.makeRuledSurface(curve.Shape, cloned_object.Shape)
        offset0 = surface.makeOffsetShape(offsetSize, 0.1, fill = False)
        offset = offset0.makeOffsetShape(-2*offsetSize, 0.1, fill = True)
        Part.show(offset)
        models.append(offset)
        doc.removeObject(curve.Name)
        doc.removeObject(feature.Name)
        doc.removeObject(cloned_object.Name)

    return models, connectorPoints

def addConnectors(connectorPoints, bridges):
    sliceNames = []
    shapeNames = []
    shapeNum = 0

    for index, points in enumerate(connectorPoints):
        templates = []
        for point, vector in points:
            template = Mesh.Mesh("C:/Users/chiru/AppData/Roaming/FreeCAD/Macro/template.stl")
            normal = normalize(getNormal(r1,r2,r3, point))
            cross_product1 = normalize(np.cross(normal, normalize(vector)))
            cross_product2 = normalize(np.cross(normal, cross_product1))
            matrix = np.column_stack((cross_product1, cross_product2, normal))

            quat = R.from_matrix(matrix).as_quat()
            template.Placement = App.Placement(App.Vector(point[0], point[1], point[2]),  App.Rotation(quat[0], quat[1], quat[2], quat[3]))
            Mesh.show(template)
            mesh_obj = doc.ActiveObject
            shape = Part.Shape()
            shape.makeShapeFromMesh(template.Topology, 0.1)
            doc.removeObject(mesh_obj.Name)
            shape_obj = Part.show(shape)
            templates.append(shape_obj)

        bridge = bridges[index]
        bridge_obj = Part.show(bridge) 

        f = BOPTools.SplitFeatures.makeSlice(name='Slice')
        f.Base = bridge_obj
        f.Tools = templates
        f.Mode = 'Split'
        f.Proxy.execute(f)
        f.purgeTouched()
        for obj in f.ViewObject.Proxy.claimChildren():
            obj.ViewObject.hide()
        CompoundTools.Explode.explodeCompound(f)
        f.ViewObject.hide()

        sliceNumText = f"{shapeNum:03d}"
        sliceNumText = sliceNumText if sliceNumText != "000" else ""

        shapeNames.append(f"Shape{sliceNumText}")
        sliceNames.append([f"Slice{sliceNumText}_child{i}" for i in range(len(templates)+1)])
        shapeNum+=1
    return sliceNames, shapeNames

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

doc = App.newDocument()
ellipsoid = doc.addObject("Part::Ellipsoid", "myEllipsoid")
ellipsoid.Radius1 = r3
ellipsoid.Radius2 = r1
ellipsoid.Radius3 = r2

dr=2
r1+=dr
r2+=dr
r3+=dr

points_coordinates = getCoordinates(r1, r2, r3)
points_mesh = renderSocktes(r1,r2,r3, points_coordinates)
bridge_models, connector_points = renderBridges(r1,r2,r3, points_coordinates)
sliceNames, shapeNames = addConnectors(connector_points, bridge_models)


for name in shapeNames:
    doc.removeObject(name)

doc.recompute()

usedSockets = {}
for names, points in zip(sliceNames, getConnectedPoints()):
    socketNames = [points[0], points[1]]
    objects =  [doc.getObject(name) for name in names]
    coords = [obj.Shape.CenterOfMass for obj in objects]
    from_coord = points_coordinates[points[0]]
    to_coord = points_coordinates[points[1]]
    pairs = list(itertools.combinations(coords, 2))
    distances = [math.dist(p1,p2) for (p1,p2) in pairs]
    if len(names) == 3:
        max_dist = max(distances)
        if distances[0] == max_dist:
            bridge = objects[2]
            objects.pop(2)
            coords.pop(2)
        elif distances[1] == max_dist:
            bridge = objects[1]
            objects.pop(1)
            coords.pop(1)
        elif distances[2] == max_dist:
            bridge = objects[0]
            objects.pop(0)
            coords.pop(0)
        Mesh.export([bridge], f"C:\\Users\\chiru\\AppData\\Roaming\\FreeCAD\\Macro\\Export\\{socketNames[0]}_{socketNames[1]}.stl")
        
    if math.dist(coords[0], from_coord) <  math.dist(coords[0], to_coord):
        objects_for_sockets = objects
    else:
        objects_for_sockets = list(reversed(objects))

        
    for i, name in enumerate(socketNames):
        if name in usedSockets:
            usedSockets[name].append(objects_for_sockets[i])
        else:
            socket = doc.getObject(name)
            usedSockets[name] = [socket, objects_for_sockets[i]]

for name in usedSockets:
    Mesh.export(usedSockets[name], f"C:\\Users\\chiru\\AppData\\Roaming\\FreeCAD\\Macro\\Export\\{name}.stl")    