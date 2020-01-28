import vtk
import torch
from models.networks import MeshEncoderDecoder, init_weights
from models.layers.mesh import Mesh

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
    # mapper.SetScalarRange([0.0, 15.0])

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)


    return actor



if __name__ == "__main__":


    #Read polydata
    reader = vtk.vtkOBJReader()
    reader.SetFileName('./data/adobe__MaleFitA_tri_fixed.obj')
    reader.Update()

    polydata = reader.GetOutput()
    print(polydata.GetNumberOfPoints())

    edgeExtractor = vtk.vtkExtractEdges()
    edgeExtractor.SetInputData(polydata)
    edgeExtractor.Update()

    lines = edgeExtractor.GetOutput().GetLines()

    print("Number of edges : ", lines.GetNumberOfCells())

    # #Copute normal
    # normalGenerator = vtk.vtkPolyDataNormals()
    # normalGenerator.SetInputData(polydata)
    # normalGenerator.ComputePointNormalsOn()
    # normalGenerator.ComputeCellNormalsOn()
    # normalGenerator.SetSplitting(False)
    # normalGenerator.Update()
    # polydata = normalGenerator.GetOutput()
    # print(polydata.GetNumberOfPoints())
    # exit()


    #Network parameter
    down_convs = [5, 64, 128, 256]
    up_convs = [256, 128, 64, 8]
    pool_res = [2250, 1350, 600]
    resblocks =  3
    net = MeshEncoderDecoder(pool_res, down_convs, up_convs, resblocks)
    init_weights(net, 'normal', 0.02)

    #Import sample mesh
    mesh = Mesh(polydata)
    mesh_feature = mesh.extract_features()
    mesh_gemm = mesh.extract_gemm_edges()
    print(mesh_gemm.shape)
    

    sample_input = torch.tensor(mesh_feature).unsqueeze(0).float()
    print(sample_input.size())
    
    y = net.forward(sample_input, [mesh])

    print(y.size())


    

    actor = make_actor(polydata)


    renderer.AddActor(actor)
    renWin.Render()
    iren.Start()