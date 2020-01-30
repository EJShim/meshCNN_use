import vtk


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
    normals = polydata.GetCellData().GetArray("Normals")

    for idx in range(numFaces):
        face = polydata.GetCell(idx)

        faceIds = [face.GetPointId(0),face.GetPointId(1), face.GetPointId(2)]
        print(normals.GetTuple(idx), faceIds)




    #Edge-based processing
    edgeGenerator = vtk.vtkExtractEdges()
    edgeGenerator.SetInputData(polydata)
    edgeGenerator.Update()

    edgePolyData = edgeGenerator.GetOutput()
    numEdges = edgePolyData.GetNumberOfCells()

    for idx in range(numEdges):
        edge = edgePolyData.GetCell(idx)

        edgeIds = [edge.GetPointId(0), edge.GetPointId(1)]
        print(edgeIds)


    


    actor = make_actor(polydata)

    renderer.AddActor(actor)

    renWin.Render()

    iren.Start()


