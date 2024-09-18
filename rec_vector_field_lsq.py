from numpy import array, zeros, sort, sign
from numpy.linalg import lstsq, det
from scipy.integrate import quad
from pydec.mesh.simplex import simplex
from pydec.dec.cochain import d, star

def integral_edge(F, p1, p2):
    r = lambda t: p2*t + (1-t)*p1
    dr = p2 - p1
    integrando = lambda t: F(r(t))@dr #pullback
    return quad(integrando, 0,1, epsabs=1e-8, epsrel=1e-8)[0]

def my_combinations(iterable):
    length = len(iterable)
    return [[iterable[i], iterable[(i + 1) % length]] for i in range(length)]

def create_edge_to_triangles(sc, triangles):
    edge_to_triangles = {i: [] for i in range(len(sc[1].simplices))}
    for i, triangle in enumerate(triangles):
        for edge in my_combinations(triangle):
            edge_index = sc[1].simplex_to_index[simplex(tuple(edge))]
            edge_to_triangles[edge_index].append(i)
    return edge_to_triangles

def get_neighbors_2level(sc, cochain, edge_index, vertices, triangles, edges_to_triangles):
        
    """
    This function returns the neighbors of a given edge in a 2-level stencil.
    It uses the edge_to_triangles dictionary to find the triangles that contain the edge.
    Then, it finds the edges informations that are shared by the triangles.
    """
    def get_edge_info(e, parity):
        v0, v1 = vertices[sc[1].simplices[e]]
        edge_vector = (v1 - v0)*(-1)**parity #parity is 0 or 1
        unit_tangent_vector = edge_vector
        midpoint = sc[1].circumcenter[e]
        value = cochain[e]*(-1)**parity
        return (tuple(midpoint), value, tuple(unit_tangent_vector))

    # Find adjacent triangles to the original edge
    result = set()  # Use a set to avoid duplicates
    edges_to_process = {edge_index}
    processed_edges = set()
    flag = True

    while edges_to_process:
        current_edge = edges_to_process.pop()
        processed_edges.add(current_edge)
        # Find triangles adjacent to the current edge
        current_parents_index = edges_to_triangles[current_edge]

        for t in current_parents_index:  
            s = triangles[t]
            parities = array([simplex(tuple(e)).parity for e in my_combinations(s)])
            edges = [(sc[1].simplex_to_index[simplex(tuple(e))], parities[i]) for i, e in enumerate(my_combinations(s))]
            for e, parity in edges:
                if e not in processed_edges:
                    result.add(get_edge_info(e, parity))
                    processed_edges.add(e)  # Add the edge to processed_edges
                    if flag:  # Limit to prevent excessive growth
                        edges_to_process.add(e)
        flag = False
    result.add(get_edge_info(edge_index, 0))
    return list(result)

def rec_vector_field_lsq(sc, cochain, edge_index, vertices, triangles, edges_to_triangles, verbose=True, hodge=False):
    
    """
    Reconstructs a vector field using least squares on a 2-level stencil.

    Parameters:
    sc (SimplicialComplex from pydec): The simplicial complex containing the mesh data.
    cochain (array): The cochain data associated with the mesh.
    edge_index (int): The index of the edge which is the center of the stencil.
    vertices (list): The vertices of the mesh.
    triangles (list): The triangles in the mesh.
    verbose (bool): If True (default), prints additional information during the process.
    
    Returns:
    function: A function that takes a point p and returns the reconstructed vector at that point.
    """

    aux = get_neighbors_2level(sc, cochain, edge_index, vertices, triangles, edges_to_triangles)
    midpoints, values, unit_tangent_vectors = zip(*aux)
    midpoints = array(midpoints)
    values = array(values)
    unit_tangent_vectors = array(unit_tangent_vectors)
    n = len(midpoints)
    
    # Construct the A matrix
    A = zeros((n, 6))
    for i in range(n):
        x, y = midpoints[i]
        nx, ny = unit_tangent_vectors[i]
        A[i] = [nx, ny, nx*x, nx*y, ny*x, ny*y]
        
    # The right-hand side is the values vector
    b = values
    # Solve the least squares problem
    coefs, residuals, rank, s = lstsq(A, b, rcond=None)
    if verbose:
        print("------Solving linear system-----")
        print(f"Residuals: {residuals}")
        print(f"Rank: {rank}")
        print(f"condition number: {s[0]/s[-1]}")
        print('----------------------------------')
    
    if hodge:
        def reconstructed_vector(p):
            x, y = p
            return array([-coefs[1] - coefs[4]*x - coefs[5]*y, 
                          coefs[0] + coefs[2]*x + coefs[3]*y])
     
        return reconstructed_vector, coefs
    
    else:
        def reconstructed_vector(p):
            x, y = p
            return array([coefs[0] + coefs[2]*x + coefs[3]*y,
                            coefs[1] + coefs[4]*x + coefs[5]*y])

        return reconstructed_vector, coefs
    
def hodge_lsq(sc, cochain, vertices, triangles, edges_to_triangles):
    """
    Computes the discrete Hodge of a 1-form over edges using least squares on a 2-level stencil.

    Parameters:
    sc (SimplicialComplex from pydec): The simplicial complex containing the mesh data.
    cochain (array): The cochain data associated with the mesh.
    vertices (list): The vertices of the mesh.
    triangles (list): The triangles in the mesh.

    Returns:
    array: The Hodge of the 1-form.
    """
    
    N1 = len(sc[1].simplices)
    hodge = zeros(N1)
    for i in range(N1):
        index = [set(sc[1].index_to_simplex[i]).issubset(t) for t in triangles]
        v0,v1 = vertices[sc[1].simplices[i]]
        parents = list(triangles[index])
        parents_index = list(sort([sc[2].simplex_to_index[simplex(t)] for t in parents]))
        edge_centroid = array(sc[1].circumcenter[i])
        rec_vec,_ = rec_vector_field_lsq(sc, cochain, i, vertices, triangles, edges_to_triangles, verbose=False, hodge=True)
        
        if len(parents_index) == 2: #Arestas internas
            c = array([sc[2].circumcenter[j] for j in parents_index])
            aux = 0
            for k,_ in enumerate(parents_index):
                circ = c[k]
                sinal = sign(det([v1-v0, edge_centroid-circ]))
                aux += sinal * integral_edge(rec_vec, circ, edge_centroid)
            hodge[i] = aux
            
        elif len(parents_index) == 1: #Arestas do bordo
            c = array([sc[2].circumcenter[j] for j in parents_index])
            c0 = c[0]
            sinal = sign(det([v1-v0, c0 - edge_centroid]))
            hodge[i] = sinal * integral_edge(rec_vec, edge_centroid, c0) 

    return hodge

def laplacian_lsq(sc, cochain, vertices, triangles, edges_to_triangles):
    """
    Computes the discrete Laplacian of a 0-form over vertices using least squares on a 2-level stencil.

    Parameters:
    sc (SimplicialComplex from pydec): The simplicial complex containing the mesh data.
    cochain (array): The 0-cochain data associated with the mesh vertices.
    vertices (list): The vertices of the mesh.
    triangles (list): The triangles in the mesh.

    Returns:
    array: The Laplacian of the discrete 0-form.
    """
    w = sc.get_cochain(1)

    #Taking the exterior derivative of the 0-cochain
    w[:] = sc[0].d @ cochain[:]

    #Computing the Hodge of the 1-form
    star_w = hodge_lsq(sc, w, vertices, triangles, edges_to_triangles)

    #Taking the Laplacian of the 1-form
    lap =  -sc[0].star_inv @ sc[0].d.T @ star_w
    return lap
    