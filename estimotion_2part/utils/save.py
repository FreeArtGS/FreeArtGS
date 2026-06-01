import os
import json


def save_point_trajectory_dict(point_trajectory_dict, output_dir):
# Convert point_trajectory_dict for JSON serialization

    json_point_trajectory_dict = {}
    for point_key, point_data in point_trajectory_dict.items():
        y_idx, x_idx = point_key
        key_str = f"{int(y_idx)},{int(x_idx)}"  # Convert tuple key to string for JSON
        json_point_trajectory_dict[key_str] = {
            "trajectories": [traj.tolist() for traj in point_data["trajectories"]],
            "pair_ids": point_data["pair_ids"],
            "segmentation_weights": point_data["segmentation_weights"],
            "point_indices": [int(y_idx), int(x_idx)]  # Store pixel coordinates as list
        }
    # Save point_trajectory_dict as JSON
    
    from utils.convert import convert_dict_to_json_format as convert_dict
    trajectory_file = os.path.join(output_dir, 'trajs.json')
    with open(trajectory_file, 'w') as f:
        # json.dump(json_point_trajectory_dict, f, indent=2)
        json.dump(convert_dict(json_point_trajectory_dict), f, indent=2)