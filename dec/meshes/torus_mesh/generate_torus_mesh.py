import gmsh

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("Torus2D")

# Define the rectangle domain [0,1] x [0,1]
Lx, Ly = 1.0, 1.0

# Define the corner points
p1 = gmsh.model.geo.addPoint(0, 0, 0)
p2 = gmsh.model.geo.addPoint(Lx, 0, 0)
p3 = gmsh.model.geo.addPoint(Lx, Ly, 0)
p4 = gmsh.model.geo.addPoint(0, Ly, 0)

# Define the lines for the boundary of the rectangle
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

# Create a curve loop and plane surface for meshing
cloop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
gmsh.model.geo.addPlaneSurface([cloop])

# Synchronize to register the geometry
gmsh.model.geo.synchronize()

# Define affine transformations to maintain orientation
# For x periodicity (shifting in x direction)
translation_x = [1, 0, 0, Lx,  # X direction remains the same, translation by Lx
                 0, 1, 0, 0,   # Y direction remains the same
                 0, 0, 1, 0,   # Z direction remains the same (no Z)
                 0, 0, 0, 1]   # Homogeneous coordinate

# For y periodicity (shifting in y direction)
translation_y = [1, 0, 0, 0,   # X direction remains the same
                 0, 1, 0, Ly,  # Y direction translated by Ly
                 0, 0, 1, 0,   # Z direction remains the same (no Z)
                 0, 0, 0, 1]   # Homogeneous coordinate

# Set periodic boundary conditions using setPeriodic
# Identify x = 0 with x = 1 (lines l1 and l3)
gmsh.model.mesh.setPeriodic(1, [l1], [l3], translation_x)

# Identify y = 0 with y = 1 (lines l4 and l2)
gmsh.model.mesh.setPeriodic(1, [l4], [l2], translation_y)

# Generate the mesh
gmsh.model.mesh.generate(2)

node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
with open("vertices.txt", "w") as vertex_file:
    for i in range(len(node_tags)):
        vertex_file.write(f"{node_coords[3*i]} {node_coords[3*i+1]}\n")

# Retrieve the triangular elements and save them to a text file
element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(2)  # 2 for surface elements

# Gmsh might generate elements other than triangles, so we ensure we're working with triangular elements
for elem_type, elems, nodes in zip(element_types, element_tags, element_node_tags):
    if elem_type == 2:  # Element type 2 corresponds to triangles
        with open("triangles.txt", "w") as triangle_file:
            for i in range(len(elems)):
                triangle_file.write(f"{nodes[3*i]} {nodes[3*i+1]} {nodes[3*i+2]}\n")


# Save the mesh to a file (optional)
gmsh.write("torus_2d.msh")

# Finalize Gmsh
gmsh.finalize()


