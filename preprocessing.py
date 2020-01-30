import vtk
import math
import numpy as np


# Renderer
renderer = vtk.vtkRenderer()

# Render window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
renWin.SetSize(1000, 1000)

# Render window interactor
iren = vtk.vtkRenderWindowInteractor()

interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)
iren.SetRenderWindow(renWin)



def make_actor(polydata):
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange([0.0, 8.0])


    actor = vtk.vtkActor()
    actor.SetMapper(mapper)


    return actor


def findNBFaces(polydata, edge):
    result = []


    #Face-based Processing
    numFaces = polydata.GetNumberOfCells()
    #normals = polydata.GetCellData().GetArray("Normals")

    for faceId in range(numFaces):
        face = polydata.GetCell(faceId)
        pointIds = [face.GetPointId(0),face.GetPointId(1), face.GetPointId(2)]

        if edge[0] in pointIds and edge[1] in pointIds:
            result.append(faceId)

    return result

if __name__ == "__main__":
    #Read polydata
    reader = vtk.vtkOBJReader()
    reader.SetFileName('./data/adobe__MaleFitA_tri_fixed.obj')
    reader.Update()
    polydata = reader.GetOutput()

    #Compute Normal
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputData(polydata)
    normalGenerator.ComputePointNormalsOff()
    normalGenerator.ComputeCellNormalsOn()
    normalGenerator.SplittingOff()
    normalGenerator.Update()
    polydata = normalGenerator.GetOutput()

    
    #Face-based Processing
    numFaces = polydata.GetNumberOfCells()

    #Initialize Edge with face iterations
    edgeData = dict()
    for faceId in range(numFaces):
        face = polydata.GetCell(faceId)
        pointIds = sorted([face.GetPointId(0),face.GetPointId(1), face.GetPointId(2)])
        #extract edges
        edges = [
            tuple([pointIds[0], pointIds[1]]),
            tuple([pointIds[0], pointIds[2]]),
            tuple([pointIds[1], pointIds[2]])
        ]
        #Append Edge!
        for edge in edges:
            if edge not in edgeData:
                #define Edge [faceid, faceid]
                edgeData[edge] = [faceId, -1]
            else:
                #Append Edge Information
                edgeData[edge][1] = faceId


    dihedral = []
    normals = polydata.GetCellData().GetArray("Normals")
    #for
    for edge in edgeData:
        faceIds = edgeData[edge]
        
        face0 = polydata.GetCell(faceIds[0])
        face1 = polydata.GetCell(faceIds[1])

        faceNormal0 = normals.GetTuple(faceIds[0])
        faceNormal1 = normals.GetTuple(faceIds[1])

        #Compute dihedral
        angle = vtk.vtkMath.Dot(faceNormal0, faceNormal1)
        if angle > 1.0 : angle = 1.0        
        angle = vtk.vtkMath.Pi() - math.acos(angle)
        dihedral.append(angle)



        print(face0.GetPointId(0), face0.GetPointId(1))


    dihedral = np.array(dihedral)


    print(dihedral)

        






    


    actor = make_actor(polydata)

    renderer.AddActor(actor)

    renWin.Render()

    iren.Start()


