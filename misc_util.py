import os
import pandas as pd
from scipy.spatial.transform import Rotation
import numpy as np

def find_best_match(df, target_value, original_df):
    # Calculate the absolute difference between each element and the target value
    diff = np.abs(df - target_value)

    # Find the index of the minimum difference
    idx = diff.values.argmin()

    # Convert the flat index to 2D index

    # row, col = np.unravel_index(idx, df.shape)
    row = idx

    # Get the actual value at this location
    best_match_value = df.iloc[row]
    found_pose = original_df.iloc[row][['tx_odometry_device', 'ty_odometry_device', 'tz_odometry_device',
                                        'qx_odometry_device', 'qy_odometry_device', 'qz_odometry_device',
                                        'qw_odometry_device']]

    return row, best_match_value, found_pose

def get_cam_wrt_world_from_olcsv(episode_timesteps, takename):
    open_loop_csv = pd.read_csv(os.path.join("/data/pulkitag/models/rli14/data/egoexo/takes", takename, "trajectory/open_loop_trajectory.csv"))

    # convert the pandas column to be in deciseconds, because we are 10fps
    # start: microseconds -> deciseconds
    tracking_times_in_deciseconds = (open_loop_csv['tracking_timestamp_us'] - open_loop_csv['tracking_timestamp_us'][0])/100000

    # note: throughout this code we are effectively indexing one below the num timesteps, which is good for robustness
    cam_poses_in_world_temp = []
    for timestep in range(episode_timesteps):
        out = find_best_match(tracking_times_in_deciseconds, timestep, open_loop_csv)

        world_pose = np.array(out[-1])

        trans = world_pose[:3]

        xyzw = world_pose[3:]
        # convert to a homogeneous matrix

        rotmat = Rotation.from_quat(xyzw).as_matrix()

        cam_poses_in_world_temp.append((trans, rotmat))
    return cam_poses_in_world_temp