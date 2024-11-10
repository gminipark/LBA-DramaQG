def get_info_from_vid(vid):

    episode_id = vid[13:15]
    scene_id = vid[16:19]
    shot_id = vid[20:]

    return (episode_id, scene_id, shot_id)