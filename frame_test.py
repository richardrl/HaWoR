import lib.vis.viewer as viewer_utils
from lib.vis.run_vis2 import camera_marker_geometry
from aitviewer.scene.camera import OpenCVCamera
import numpy as np
from misc_util import get_cam_wrt_world_from_olcsv
import os
import torch
from scipy.spatial.transform import Rotation

# load stuff
seq_folder = "/home/rli14/Desktop/minnesota_cooking_074_2"
print("ln10")
camposes = get_cam_wrt_world_from_olcsv(10000, os.path.basename(seq_folder))
print("ln12")

# R_c2w_sla_all = torch.from_numpy(np.asarray([_[1] for _ in camposes]).copy()).to(torch.float32)
# t_c2w_sla_all = torch.from_numpy(np.stack([_[0] for _ in camposes]).copy().astype(np.float32)).to(torch.float32)
R_c2w_sla_all = np.asarray([_[1] for _ in camposes]).copy()
t_c2w_sla_all = np.stack([_[0] for _ in camposes]).copy().astype(np.float32)
# end load stuff


verts, faces, face_colors = camera_marker_geometry(0.05, 0.1)

# produce the sequence data
# T, # camera vertices = 5, 3
verts = np.einsum("tij,nj->tni", R_c2w_sla_all, verts) + t_c2w_sla_all[:, np.newaxis, ...]

camera_meshes = {
    "v3d": verts,
    "f3d": faces,
    "vc": None,
    "name": "camera",
    "fc": face_colors,
    "color": -1,
}
vis_dict = dict()
vis_dict["camera"] = camera_meshes

meshes = viewer_utils.construct_viewer_meshes(
    vis_dict, draw_edges=False, flat_shading=False
)


vis_w = 512
vis_h = 512
viewer = viewer_utils.ARCTICViewer(interactive=True, size=(vis_w, vis_h))
for mesh in meshes.values():
    viewer.v.scene.add(mesh)

# add opencv camera

fov = 60
f = 606
cam_intrinsics = np.array([[f, 0.0, 256], [0.0, f, 256], [0.0, 0.0, 1.0]])
# cam_extrincs = np.eye(4, 4)[:3, :]
cam_extrinsics_play = np.tile(np.eye(4, 4), (10000, 1, 1))
cam_extrinsics_play[:, :3, :3] = R_c2w_sla_all
cam_extrinsics_play[:, :3, 3] = t_c2w_sla_all

bodyframe_rotation = Rotation.from_euler("Z", 180, degrees=True).as_matrix()
homo_rot = np.eye(4)
homo_rot[:3, :3] = bodyframe_rotation
cam_extrinsics_play = cam_extrinsics_play @ homo_rot

cam_extrinsics_play = np.linalg.inv(cam_extrinsics_play)

cameras = OpenCVCamera(cam_intrinsics, cam_extrinsics_play[:, :3, :], 512, 512, viewer=viewer)

viewer.v.scene.add(cameras)

viewer.view_interactive()