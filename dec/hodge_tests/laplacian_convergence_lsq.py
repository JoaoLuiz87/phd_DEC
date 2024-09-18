import numpy as np
from numpy import zeros, sort, loadtxt, array, pi, sin, cos, log, sqrt, sum
import matplotlib.pyplot as plt
from numpy.linalg import norm
from pydec import simplicial_complex
from pydec.mesh.simplex import simplex
from pydec.mesh.subdivision import loop_subdivision
from scipy.integrate import quad
from rec_vector_field_lsq import *
from pandas import DataFrame

def integral_edge(F,p1,p2):
    r = lambda t: p2*t + (1-t)*p1
    dr = p2 - p1
    integrando = lambda t: F(r(t))@dr #pullback
    return quad(integrando, 0,1, epsabs=1e-13, epsrel=1e-13)[0]
def order(le, lh):
    p = [(log(le[i]/le[i+1]))/(log(lh[i]/lh[i+1])) for i in range(len(le)-1)]
    if p == []:
        return None
    return p[-1]

#----------------- Funções para teste ----------------------------
u1 = lambda x,y: np.cos(pi*x)*np.cos(pi*y)
lap_u1 = lambda x,y: -2*pi**2*np.cos(pi*x)*np.cos(pi*y)

u2 = lambda x,y: np.cos(x*y)
lap_u2 = lambda x,y: -(x**2 + y**2)*np.cos(x*y)

u3 = lambda x,y: np.sin(y)*x**2
lap_u3 = lambda x,y: 2*np.sin(y) - np.sin(y)*x**2

u4 = lambda x,y: x**2 + y**2
lap_u4 = lambda x,y:  4*np.ones(len(x))

u5 = lambda x,y: (1/6)*x**3 + y**2
lap_u5 = lambda x,y:  x + 2.0

u_list = [u1, u2, u3, u4, u5]
lap_list = [lap_u1, lap_u2, lap_u3, lap_u4, lap_u5]
#------------------------------------------------------------------
#COMEÇO DOS TESTES    

num_loops = 4

meshes = ['square_mesh', 'tilted_mesh','circular_mesh']
vertices = loadtxt('../meshes/0.5_square_mesh/vertices.txt')
triangles = loadtxt('../meshes/0.5_square_mesh/triangles.txt', dtype='int') - 1
sc = simplicial_complex((vertices,triangles))

for mesh in meshes:
    results = []
    if mesh=='square_mesh':
        vertices = loadtxt('../meshes/0.5_square_mesh/vertices.txt')
        triangles = loadtxt('../meshes/0.5_square_mesh/triangles.txt', dtype='int') - 1
        sc = simplicial_complex((vertices,triangles))
    elif mesh=='tilted_mesh':
        vertices = array([[0,0],[1,0],[0.5,1],[1.5,1.0]])
        triangles = array([[0,1,2],[1,3,2]], dtype='int')
        vertices, triangles = loop_subdivision(vertices, triangles)
        sc = simplicial_complex((vertices,triangles))
    elif mesh=='circular_mesh':
        vertices = loadtxt('../meshes/circular_mesh/vertices.txt')
        triangles = loadtxt('../meshes/circular_mesh/triangles.txt', dtype='int') - 1
        sc = simplicial_complex((vertices,triangles))
    
    print('Tests for mesh '+mesh)
    for i, (u, lap_u) in enumerate(zip(u_list, lap_list)):
        print(f"Processing function {i+1}/{len(u_list)}")
        
        vertices_current = vertices.copy()
        triangles_current = triangles.copy()
        sc_current = sc
        lap_error_l2_list = []
        lap_error_inf_list = []
        max_volume_list = [max(sc_current[1].primal_volume)]
        for j in range(num_loops):
            print(f'  Performing loop {j+1}/{num_loops}, num_vertices = {len(vertices_current)}')
            
            cochain = u(vertices_current[:,0], vertices_current[:,1])
            print(len(cochain))
            print(len(sc_current[0].simplices))
            true_lap = lap_u(vertices_current[:,0], vertices_current[:,1])

            edges_to_triangles = create_edge_to_triangles(sc_current, triangles_current)
            approx_lap = laplacian_lsq(sc_current, cochain, vertices_current, triangles_current, edges_to_triangles)
            
            x_points = set([(e[0]) for e in sc_current.boundary()])
            y_points = set([(e[1]) for e in sc_current.boundary()])
            boundary_points_index = sort(list(x_points.union(y_points)))
            boundary_points = vertices_current[boundary_points_index]

            internal_points = set(sc_current[0].simplex_to_index) - set([sc_current[0].index_to_simplex[e] for e in boundary_points_index])
            internal_index = sort(list(sc_current[0].simplex_to_index[p] for p in internal_points))
        
            # L2 error
            l2_error = norm((array(approx_lap[internal_index]) - array(true_lap[internal_index])) * sc_current[0].dual_volume[internal_index]**(1/2))
            
            # L-inf error
            inf_error = max(abs(approx_lap[internal_index] - true_lap[internal_index]))
            
            lap_error_l2_list.append(l2_error)
            lap_error_inf_list.append(inf_error)
            ordem_l2 = order(lap_error_l2_list, max_volume_list)
            ordem_inf = order(lap_error_inf_list, max_volume_list)
            results.append({
                'Function': f'u{i+1}',
                'Num Vertices': len(vertices_current),
                'Num Triangles': len(triangles_current),
                'Max Primal Volume': max_volume_list[-1],
                'L2 Error': l2_error,
                'Inf Error': inf_error,
                'Order L2': ordem_l2,
                'Order inf': ordem_inf
            })
            
            vertices_current, triangles_current = loop_subdivision(vertices_current, triangles_current)
            sc_current = simplicial_complex((vertices_current, triangles_current))
            max_volume_list.append(max(sc_current[1].primal_volume))
    df = DataFrame(results)

    # Save to CSV
    df.to_csv('lap_convergence_'+mesh+'.csv', index=False)
    print("Results saved for "+mesh)