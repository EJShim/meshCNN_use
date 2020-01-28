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
    mapper.SetScalarRange([1.0, 8.0])

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)


    return actor

def read_gt(filepath):
    print(filepath)

    result = []

    with open(filepath) as f:
        content = f.readlines()
    
    result = [int(x.strip()) for x in content]


    return result

def assign_gt(polydata, gt):

    cellColor = vtk.vtkUnsignedCharArray()
    cellColor.SetNumberOfComponents(1)
    cellColor.SetNumberOfTuples(polydata.GetNumberOfCells())
    cellColor.SetName("gt")

    for i in range(polydata.GetNumberOfCells()):
        cellColor.SetTuple1(i, gt[i])
    
    polydata.GetCellData().SetScalars(cellColor)

    return polydata


if __name__ == "__main__":


    #Read polydata
    reader = vtk.vtkOBJReader()
    reader.SetFileName('./data/adobe__MaleFitA_tri_fixed.obj')
    reader.Update()
    polydata = reader.GetOutput()

    edgeExtractor = vtk.vtkExtractEdges()
    edgeExtractor.SetInputData(polydata)
    edgeExtractor.Update()

    edgePoly = edgeExtractor.GetOutput()
    #read groundtruth
    groundtruth = read_gt('./data/adobe__MaleFitA_tri_fixed.eseg')

    edgePoly = assign_gt(edgePoly, groundtruth)
    
    actor = make_actor(edgePoly)


    renderer.AddActor(actor)

    renWin.Render()

    iren.Start()

