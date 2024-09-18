import gmsh
import math

# Initialize Gmsh
gmsh.initialize()

# Create a new model
gmsh.model.add("disk_mesh")

# Define the parameters of the disk
radius = 1.0
center_x = 0.0
center_y = 0.0
center_z = 0.0

# Add a point at the center
center = gmsh.model.geo.addPoint(center_x, center_y, center_z)

# Add points around the boundary of the disk
n_boundary_points = 50  # Number of points along the boundary
boundary_points = []
for i in range(n_boundary_points):
    angle = 2 * math.pi * i / n_boundary_points
    x = center_x + radius * math.cos(angle)
    y = center_y + radius * math.sin(angle)
    z = center_z
    boundary_points.append(gmsh.model.geo.addPoint(x, y, z))

# Create a circle by connecting the boundary points
boundary_lines = []
for i in range(n_boundary_points):
    next_point = (i + 1) % n_boundary_points
    boundary_lines.append(gmsh.model.geo.addLine(boundary_points[i], boundary_points[next_point]))

# Create a closed curve (loop) from the boundary lines
loop = gmsh.model.geo.addCurveLoop(boundary_lines)

# Create a plane surface enclosed by the loop
surface = gmsh.model.geo.addPlaneSurface([loop])

# Synchronize to process the geometry
gmsh.model.geo.synchronize()

# Define the mesh size
mesh_size = 1.0
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

# Generate the mesh
gmsh.model.mesh.generate(2)

# Retrieve the vertices (nodes) and save them to a text file
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

# Optionally save the mesh to a file
gmsh.write("disk_mesh.msh")

# Launch the GUI to visualize (optional)
gmsh.fltk.run()

# Finalize the Gmsh API
gmsh.finalize()
