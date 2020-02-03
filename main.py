import vtk
import numpy as np
import torch
from models.networks import MeshEncoderDecoder, init_weights
from models.layers.mesh import Mesh
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


def assign_prediction(polydata, predict):
    #Make Lien Polydata

    edgeExtractor = vtk.vtkExtractEdges()
    edgeExtractor.SetInputData(polydata)
    edgeExtractor.Update()

    edgePoly = edgeExtractor.GetOutput()
    
    
    #Assign Ground Truth
    cellColor = vtk.vtkUnsignedCharArray()
    cellColor.SetNumberOfComponents(1)
    cellColor.SetNumberOfTuples(edgePoly.GetNumberOfCells())
    cellColor.SetName("gt")

    for i in range(edgePoly.GetNumberOfCells()):
        #print(predict[i].item())
        cellColor.SetTuple1(i, predict[i].item())
    
    edgePoly.GetCellData().SetScalars(cellColor)


    return edgePoly


    



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
    #FileName = './data/Alien Animal_ver4.obj'
    FileName = './data/adobe__MaleFitA_tri_fixed.obj'
    reader.SetFileName(FileName)
    print("======= READ ", FileName, " =======")
    reader.Update()

    polydata = reader.GetOutput()


    #Network parameter
    down_convs = [5, 32, 64, 128, 256]
    up_convs = [256, 128, 64, 32, 8]
    pool_res = [2250, 1800, 1350, 600]
    resblocks =  3
    net = MeshEncoderDecoder(pool_res, down_convs, up_convs, resblocks)

    #Try to get pretrained mesh
    state_dict = torch.load('./data/latest_net.pth')
    net.load_state_dict(state_dict)
    net.eval()

    #Import sample mesh
    mesh = Mesh(polydata)
    mesh_feature = mesh.extract_features()

    # mesh_gemm = mesh.extract_gemm_edges()
    # print(mesh_gemm.shape)


    sample_input = torch.tensor(mesh_feature).unsqueeze(0).float()
    print("sample input : ", sample_input.size()) #torch.Size([1, 5, 69823)])
    y = net.forward(sample_input, [mesh])
    y_value, y_index = y.max(-2)


    #Get Predicted Class
    predict = y_index[0]
    #print(predict.size()) #torch.Size([2250])




    #Visualize
    polydata = assign_prediction(polydata, predict)
    actor = make_actor(polydata)

    renderer.AddActor(actor)
    renWin.Render()
    iren.Start()