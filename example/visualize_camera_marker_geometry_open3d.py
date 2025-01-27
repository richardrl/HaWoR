from lib.vis.run_vis2 import camera_marker_geometry


import open3d as o3d
import numpy as np


# Get geometry data
vertices, faces, face_colors = camera_marker_geometry(0.05, 0.1)

# Create mesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.vertex_colors = o3d.utility.Vector3dVector(face_colors[:, :3])

# Create coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

# Visualization
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add geometries
vis.add_geometry(mesh)
vis.add_geometry(frame)

# Set default camera view
# ctr = vis.get_view_control()
# ctr.set_zoom(0.8)
# ctr.set_lookat([0, 0, -1.0])

# Run visualization
vis.run()
vis.destroy_window()