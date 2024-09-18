import pygmsh
import numpy as np
import matplotlib.pyplot as plt

with pygmsh.geo.Geometry() as geom:
    geom.add_circle([0.0, 0.0], 1.0, mesh_size=0.5)
    mesh = geom.generate_mesh()

print(np.array(mesh.points))
print(np.array(mesh.cells_dict['triangle']))
points = np.delete(mesh.points, 2,1)
np.savetxt("vertices.txt", points, fmt="%g")
np.savetxt("triangles.txt", mesh.cells_dict['triangle'], fmt="%d")

plt.triplot(points[:,0], points[:,1], mesh.cells_dict['triangle'])
plt.show()