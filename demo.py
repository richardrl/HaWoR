import argparse
import sys
import os

import torch
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import joblib
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller

try:
    from scripts.scripts_test_video.hawor_slam import hawor_slam
except ImportError:
    print("hawor_slam not found, skipping")

from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam
import joblib

# NOTE: to disable the checkerboard, uncheck "ground" in the GUI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--seq_folder", type=str, help="Skip video decoding step and use folder of images")
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint",  type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight",  type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--vis_mode",  type=str, default='world', help='cam | world')
    parser.add_argument("--identity_slam", action="store_true", help='use identity slam poses')
    parser.add_argument('--slam_mode',
                        type=int,
                        choices=[0, 1, 2],
                        default=0,
                        help='0 run from video, 1 egoexo from takes json, 2 identity')
    parser.add_argument('--detector', default="hands23",type=str, help='yolo | hands23')
    parser.add_argument("--image_subdir", default="extracted_images", help="Use with seq_folder. Seq folder basename is always assumed to be the take name. The images might not be located directly in the take folder, so this specifies it.")
    parser.add_argument("--label_root", default=None, help="Store the labels in this root, so as to not pollute the original image folder. By default, we store HAWOR style in the image folder.")
    parser.add_argument("--hands_bbox_root_dir", default=None, help="Store the hands bbox in this root.")
    args = parser.parse_args()

    args.seq_folder = args.seq_folder.rstrip("/")

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    if args.img_focal:
        img_focal = args.img_focal

    if args.slam_mode == 0:
        # run slam
        print(f"using img focal {img_focal}")
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        if not os.path.exists(slam_path):
            hawor_slam(args, start_idx, end_idx)
        slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)
    elif args.slam_mode == 1:
        # load slam from egoexo file
        from misc_util import get_cam_wrt_world_from_olcsv
        camposes = get_cam_wrt_world_from_olcsv(len(imgfiles), os.path.basename(args.seq_folder))

        R_c2w_sla_all = torch.from_numpy(np.asarray([_[1] for _ in camposes]).copy()).to(torch.float32)
        t_c2w_sla_all = torch.from_numpy(np.stack([_[0] for _ in camposes]).copy().astype(np.float32)).to(torch.float32)
        R_w2c_sla_all = None
        t_w2c_sla_all = None

        assert not torch.isnan(R_c2w_sla_all).any()
        assert not torch.isnan(t_c2w_sla_all).any()
    else:
        print("ln39 using identity slam poses")
        R_w2c_sla_all = torch.eye(3).unsqueeze(0).expand(len(imgfiles), -1, -1)
        t_w2c_sla_all = torch.zeros(len(imgfiles), 3)

        R_c2w_sla_all = torch.eye(3).unsqueeze(0).expand(len(imgfiles), -1, -1)
        t_c2w_sla_all = torch.zeros(len(imgfiles), 3)

    # this outputs all the infilled variables
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)

    seq_name = os.path.basename(seq_folder.rstrip("/"))
    if args.label_root is None:
        label_seq_folder = seq_folder
    else:
        label_seq_folder = os.path.join(args.label_root, seq_name)

    if os.path.exists(os.path.join(label_seq_folder, "world_space_res.pth")):
        print("ln75 loading previous infill")
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(os.path.join(label_seq_folder, "world_space_res.pth"))

    hand2idx = {
        "right": 1,
        "left": 0
    }
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1
            
    # get faces
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)

    # get right hand vertices
    hand = 'right'
    hand_idx = hand2idx[hand]
    pred_glob_r = run_mano(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])
    right_verts = pred_glob_r['vertices'][0]
    right_dict = {
            'vertices': right_verts.unsqueeze(0),
            'faces': faces_right,
        }

    # get left hand vertices
    faces_left = faces_right[:,[0,2,1]]
    hand = 'left'
    hand_idx = hand2idx[hand]
    pred_glob_l = run_mano_left(pred_trans[hand_idx:hand_idx+1, vis_start:vis_end], pred_rot[hand_idx:hand_idx+1, vis_start:vis_end], pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end], betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end])

    # outputs hands in world frame
    left_verts = pred_glob_l['vertices'][0]
    left_dict = {
            'vertices': left_verts.unsqueeze(0),
            'faces': faces_left,
        }

    R_x = torch.tensor([[1,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())

    # Here we use aitviewer(https://github.com/eth-ait/aitviewer) for simple visualization.
    if args.vis_mode == 'world': 
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        run_vis2_on_video(left_dict, right_dict, output_pth, img_focal, image_names, R_c2w=R_c2w_sla_all[vis_start:vis_end], t_c2w=t_c2w_sla_all[vis_start:vis_end])
    elif args.vis_mode == 'cam':
        output_pth = os.path.join(seq_folder, f"vis_{vis_start}_{vis_end}")
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        image_names = imgfiles[vis_start:vis_end]
        print(f"vis {vis_start} to {vis_end}")
        run_vis2_on_video_cam(left_dict, right_dict, output_pth, img_focal, image_names, R_w2c=R_w2c_sla_all[vis_start:vis_end], t_w2c=t_w2c_sla_all[vis_start:vis_end])
    else:
        print("Invalid vis mode, not vis'ing")
    print("finish")



