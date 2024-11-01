import os
from utils import get_info_from_vid
from glob import glob


def get_images_from_vid(image_dir, vid):

    episode_id, scene_id, shot_id = get_info_from_vid(vid)

    if shot_id == "0000":
        image_path = os.path.join(
            image_dir, f"AnotherMissOh{episode_id}", f"{scene_id}", "**/*.jpg"
        )
        image_list = glob(image_path, recursive=True)

    else:
        image_path = os.path.join(
            image_dir, f"AnotherMissOh{episode_id}/{scene_id}/{shot_id}/*.jpg"
        )
        image_list = glob(image_path)

    return image_list
