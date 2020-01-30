import numpy as np
import os
import ntpath
import vtk


def fill_mesh(mesh, polydata):

    #Get VS and faces..
    points = polydata.GetPoints()
    numPoints = polydata.GetNumberOfPoints()
    vs = []
    for i in range(numPoints):
        vs.append(polydata.GetPoint(i))
    mesh.vs = np.array(vs)    

    #Get Face Edges
    numFaces = polydata.GetNumberOfCells()
    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    sides = []
    edge2key = dict() #edge index?????
    edges = []
    edges_count = 0
    nb_count = []




    #do not like this....
    for i in range(numFaces):
        cell = polydata.GetCell(i)
        faces_edges = []

        faces_edges.append( tuple(sorted([cell.GetPointId(0), cell.GetPointId(1)])) )
        faces_edges.append( tuple(sorted([cell.GetPointId(0), cell.GetPointId(2)])) )
        faces_edges.append( tuple(sorted([cell.GetPointId(1), cell.GetPointId(2)])) )

        # face_edges : [(7, 343), (7, 688), (343, 688)]
        
        for edge in faces_edges:            
            if edge not in edge2key:
                # edges2key = vertex 
                edge2key[edge] = edges_count # {(7, 343): 0}
                edges.append(edge) # [(7, 343)]
                edge_nb.append([-1, -1, -1, -1])
                sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                nb_count.append(0)
                edges_count += 1
        # 7 위치에 (0,1) , 343 위치에 (0,2), 688 위치에 (1,2)
        # 즉, 0이라는 edge는 (7,343) 1이라는 edge는 (7,686) 이런식이다.
        # 여기서 0,1,2, .. 이런걸 edge_key라고 부른다.
        
        
        for idx, edge in enumerate(faces_edges):
            
            edge_key = edge2key[edge]  
            # edge : (7, 343), edge2key : {(7, 343): 0, (7, 688): 1, (343, 688): 2},  edges_key = 0
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2
        
                 
            """
            edge의 고유 넘버를 가지고 접근해서 차례대로 edge_nb 에 같은 face에 있던 edge의 고유 넘버 추가하기
            [[1, 2, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
            [[1, 2, -1, -1], [2, 0, -1, -1], [-1, -1, -1, -1]]
            [[1, 2, -1, -1], [2, 0, -1, -1], [0, 1, -1, -1]]
            """
        #print(edge, edge2key, edge_key)
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
            sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2

            """
            [[1, 0, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]
            [[1, 0, -1, -1], [1, 0, -1, -1], [-1, -1, -1, -1]]
            [[1, 0, -1, -1], [1, 0, -1, -1], [1, 0, -1, -1]]
            """
            
            """
            원래 코드에서 def build_gemm 함수 참고
            gemm_edges: array (#E x 4) of the 4 one-ring neighbors for each edge
            sides: array (#E x 4) indices (values of: 0,1,2,3) indicating where an edge is in the gemm_edge entry of the 4 neighboring edges
            for example edge i -> gemm_edges[gemm_edges[i], sides[i]] == [i, i, i, i]
            """
        
        


    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    mesh.sides = np.array(sides, dtype=np.int64)
    mesh.edges_count = edges_count
    """
    for i in range(len(edge_nb)):
        print(edge_nb[i])
    exit()
    """

    """
    #이해를 돕고자 쓴 코드
    print("edge : ", mesh.edges[0]) # edge :  [  7 343]
    print("gemm : ", mesh.gemm_edges[0]) # gemm :  [  1   2 488  26]
    print("side : ", mesh.sides[0]) # side :  [1 0 1 2]
    print("edge_count : ", mesh.edges_count) # 2250
    print(mesh.gemm_edges[mesh.gemm_edges[0], mesh.sides[0]]) # [0 0 0 0]
    
    print(mesh.gemm_edges[1]) # [   2    0 1660 1659]
    print(mesh.gemm_edges[2]) # [   0    1 1551 1257]
    print(mesh.gemm_edges[488]) # [ 26   0 846 845]
    print(mesh.gemm_edges[26]) # [ 27  25   0 488]

    exit()
    """

    #Below is the importatnt
    mesh.v_mask = np.ones(len(mesh.vs), dtype=bool)
    
    
    #Try
    mesh.features = extract_features(mesh)
    
    ## 추가
    #print((mesh.features).shape) #(5, 69823)


def extract_features(mesh):
    features = []
    edge_points = get_edge_points(mesh)


    #First feature = dihderal feature
    dihedral_feature = dihedral_angle(mesh, edge_points)
    features.append(dihedral_feature)

    #Second = symmetric opposite features
    symmetric_opposite_feature = symmetric_opposite_angles(mesh, edge_points)
    features.append(symmetric_opposite_feature)

    #Third = symmetric ratios
    symmetric_ratios_feature = symmetric_ratios(mesh, edge_points)
    features.append(symmetric_ratios_feature)

    ## 추가
    # print(np.concatenate(features,axis=0).shape) #(5, 137844)

    return np.concatenate(features, axis=0)


def dihedral_angle(mesh, edge_points):
    normals_a = get_normals(mesh, edge_points, 0)
    normals_b = get_normals(mesh, edge_points, 3)
    dot = np.sum(normals_a * normals_b, axis=1).clip(-1, 1)
    angles = np.expand_dims(np.pi - np.arccos(dot), axis=0)
    return angles


def symmetric_opposite_angles(mesh, edge_points):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles_a = get_opposite_angles(mesh, edge_points, 0)
    angles_b = get_opposite_angles(mesh, edge_points, 3)
    angles = np.concatenate((np.expand_dims(angles_a, 0), np.expand_dims(angles_b, 0)), axis=0)
    angles = np.sort(angles, axis=0)
    return angles


def symmetric_ratios(mesh, edge_points):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios_a = get_ratios(mesh, edge_points, 0)
    ratios_b = get_ratios(mesh, edge_points, 3)
    ratios = np.concatenate((np.expand_dims(ratios_a, 0), np.expand_dims(ratios_b, 0)), axis=0)
    return np.sort(ratios, axis=0)


def get_edge_points(mesh):
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
    return edge_points


def get_side_points(mesh, edge_id):

    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]


def get_normals(mesh, edge_points, side):
    edge_a = mesh.vs[edge_points[:, side // 2 + 2]] - mesh.vs[edge_points[:, side // 2]]
    edge_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2]]
    normals = np.cross(edge_a, edge_b)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=0.1)
    normals /= div[:, np.newaxis]
    return normals

def get_opposite_angles(mesh, edge_points, side):
    edges_a = mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]
    edges_b = mesh.vs[edge_points[:, 1 - side // 2]] - mesh.vs[edge_points[:, side // 2 + 2]]

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def get_ratios(mesh, edge_points, side):
    edges_lengths = np.linalg.norm(mesh.vs[edge_points[:, side // 2]] - mesh.vs[edge_points[:, 1 - side // 2]],
                                   ord=2, axis=1)
    point_o = mesh.vs[edge_points[:, side // 2 + 2]]
    point_a = mesh.vs[edge_points[:, side // 2]]
    point_b = mesh.vs[edge_points[:, 1 - side // 2]]
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths

def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div
