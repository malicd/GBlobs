import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes


def main(args):
    nusc = NuScenes(
        version="v1.0-test", dataroot="../data/robosense/v1.0-test", verbose=True
    )

    with open(args.path_glob, "r") as file:
        data_xyz = json.load(file)

    with open(args.path_gblobs, "r") as file:
        data_gblobs = json.load(file)

    final_json = dict(results=dict(), meta=data_gblobs["meta"])

    for scene in tqdm(data_gblobs["results"].keys()):
        final_json["results"][scene] = []

        det_xyz = data_xyz["results"][scene]
        det_gblobs = data_gblobs["results"][scene]

        # get ego pose
        s_record = nusc.get("sample", scene)
        sample_data_token = s_record["data"]["LIDAR_TOP"]
        sd_record = nusc.get("sample_data", sample_data_token)
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
        ego_translation = pose_record["translation"]

        df_gblobs = pd.DataFrame(det_gblobs)
        df_gblobs["distance"] = df_gblobs.apply(
            lambda row: np.linalg.norm(np.array(row["translation"]) - ego_translation),
            axis=1,
        )
        df_gblobs = df_gblobs.sort_values("detection_score", ascending=False)

        df_xyz = pd.DataFrame(det_xyz)
        df_xyz["distance"] = df_xyz.apply(
            lambda row: np.linalg.norm(np.array(row["translation"]) - ego_translation),
            axis=1,
        )
        df_xyz = df_xyz.sort_values("detection_score", ascending=False)

        df_res = pd.concat(
            [
                df_gblobs[df_gblobs["distance"] <= args.thold].drop("distance", axis=1),
                df_xyz[df_xyz["distance"] >= args.thold].drop("distance", axis=1),
            ]
        )
        df_res = df_res.sort_values("detection_score", ascending=False)
        df_res = df_res[: min(df_res.shape[0], 400)]

        final_json["results"][scene].extend(df_res.to_dict("records"))

    with open("./results_nusc.json", "w") as json_file:
        json.dump(final_json, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_glob", type=str, required=True)
    parser.add_argument("--path_gblobs", type=str, required=True)
    parser.add_argument("--thold", type=float, default=20.0)

    # Parse the arguments
    args = parser.parse_args()
    main(args)
