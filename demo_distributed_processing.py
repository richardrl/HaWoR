import multiprocessing as mp
from typing import List, Dict
import subprocess
import sys
from pathlib import Path
import argparse

def run_script(args_dict: Dict) -> None:
    """Run demo.py with given arguments"""
    cmd = [sys.executable, "demo.py"]
    
    # Convert dict to command line args
    for key, value in args_dict.items():
        cmd.extend([f"--{key}", str(value)])
    
    # Run process without capturing output (will print to terminal directly)
    subprocess.run(cmd)

# def launch_parallel(args_list: List[Dict]) -> None:
#     """Launch scripts in parallel across CPUs"""
#     with mp.Pool() as pool:
#         pool.map(run_script, args_list)

if __name__ == "__main__":
    import json
    import os
    import math
    # read the seq folders from the image root then populate the label root in parallel
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--label_root", type=str, required=True)
    parser.add_argument("--hands_bbox_root_dir", type=str, required=True)
    parser.add_argument("--allowed_parent_tasks", nargs='+', type=str, help="What parent tasks to allow")
    parser.add_argument("--mp_idx", type=int, default=None, help="An index to use for distributed processing. The overall strategy is to split the seq folders to be distributed amongst all the different slurm jobs.")
    parser.add_argument("--mp_total", type=int, default=None, help="Total number of jobs to use for distributed processing.")
    args = parser.parse_args()

    # do .name to get the names
    seq_folder_paths = [f for f in Path(args.image_root).iterdir() if f.is_dir()]

    # we need to filter seq folder paths by the task name because the bboxes do not exist for all tasks
    if args.allowed_parent_tasks:
        # filter to only keep directories with the parent takes
        takes_json_lst = json.load(open(os.path.join("/data/pulkitag/models/rli14/data/egoexo", "takes.json")))
        valid_take_names = [take['take_name'] for take in takes_json_lst if
                            take['parent_task_name'] in args.allowed_parent_tasks]

        seq_folder_paths = [dir_ for dir_ in seq_folder_paths if dir_.name in valid_take_names]

    # always filter by completed
    seq_folder_paths = [dir_ for dir_ in seq_folder_paths if not os.path.exists(os.path.join(dir_, "world_space_res.pth"))]

    if args.mp_idx is not None:
        assert args.mp_total
        # split the img paths to be distributed amongst all the different slurm jobs
        # TODO: check this more carefully
        start_img_path_idx = math.floor((len(seq_folder_paths) / args.mp_total) * args.mp_idx)
        end_img_path_idx = math.floor((len(seq_folder_paths) / args.mp_total) * (args.mp_idx + 1))
        seq_folder_paths = seq_folder_paths[start_img_path_idx:end_img_path_idx]

    default_dict = dict(
        vis_mode=None,
        img_focal=150,
        slam_mode=1,
        image_subdir="",
        label_root=args.label_root,
        hands_bbox_root_dir=args.hands_bbox_root_dir,
        detector="hands23",
    )

    for seq_folder_path in seq_folder_paths:
        run_script(dict(seq_folder=seq_folder_path) | default_dict)
    # args_list = [
    #     dict(seq_folder=seq_folder_paths[idx]) | default_dict for idx in range(len(seq_folder_paths))
    # ]
    
    # launch_parallel(args_list)