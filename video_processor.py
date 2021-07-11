import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from keras.backend import set_session
import pandas as pd
from pathlib import Path
import globals

import posenet


class VideoProcessor:
    def __init__(self, pose_net_estimator, pose2kinect_estimator, kinect3d_estimator, key_frame_estimator):
        self._pose_net_estimator = pose_net_estimator
        self._pose2kinect_estimator = pose2kinect_estimator
        self._kinect3d_estimator = kinect3d_estimator
        self._key_frame_estimator = key_frame_estimator
        self._eval_freq = 5

    def process(self, video_path, csv_path, session, graph, video_id):
        input_video = NamedTemporaryFile(suffix=Path(video_path).suffix)
        with open(video_path, "rb") as from_v:
            data = from_v.read()

        with open(input_video.name, "wb") as to_v:
            to_v.write(data)

        cap = cv2.VideoCapture(input_video.name)
        frame_pause = int(cap.get(cv2.CAP_PROP_FPS) / self._eval_freq)
        all_poses_df = pd.DataFrame(data={})
        all_pose_metrics = {}

        ret, frame = cap.read()

        while ret:
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_progress = int(frame_index / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 80)
            if new_progress > globals.progress.get(video_id):
                globals.progress.update({video_id : new_progress})


            if (frame_index - 1) % frame_pause == 0:
                pose_df, pose_metrics = self._pose_net_estimator.evaluate(frame, return_raw_output=True)
                all_poses_df = pd.concat([all_poses_df, pose_df])
                all_pose_metrics.update({frame_index: pose_metrics})

            ret, frame = cap.read()
            print(f"Frame: {frame_index} from {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

        with graph.as_default():
            set_session(session)
            poses2kinect_values = self._pose2kinect_estimator.predict(all_poses_df)
            kinect3d_df = self._kinect3d_estimator.predict3d(poses2kinect_values, return_df=True)
            start_frame_i, stop_frame_i = self._key_frame_estimator.predict_video_key_frames(
                kinect3d_df, cap.get(cv2.CAP_PROP_FPS))
        frame_indexes = list(range(start_frame_i, stop_frame_i))
        kinect3d_df = kinect3d_df.iloc[frame_indexes]
        kinect3d_df["FrameNum"] = list(range(start_frame_i, stop_frame_i))
        kinect3d_df.to_csv(csv_path, index=False)

        # Write video with pose
        processed_indexes = np.array(list(all_pose_metrics.keys()))
        processed_indexes = processed_indexes[processed_indexes <= int(cap.get(cv2.CAP_PROP_POS_FRAMES))]

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_i)
        pose_metrics = all_pose_metrics[processed_indexes.max()]
        ret, frame = cap.read()
        video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"XVID"), int(cap.get(cv2.CAP_PROP_FPS)),
            (frame.shape[1], frame.shape[0]))

        while ret and stop_frame_i >= cap.get(cv2.CAP_PROP_POS_FRAMES):
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_progress = int(((frame_index - start_frame_i) / (stop_frame_i - start_frame_i) * 25) + 75)
            if new_progress > globals.progress.get(video_id):
                globals.progress.update({video_id : new_progress})

            pose_scores, keypoint_scores, keypoint_coords = pose_metrics
            frame = posenet.draw_skel_and_kp(
                frame, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.00, min_part_score=0.00)
            video_writer.write(frame)

            if frame_index in all_pose_metrics.keys():
                pose_metrics = all_pose_metrics[frame_index]
            ret, frame = cap.read()

        #     Just for testing !

        globals.results[video_id] = {"status": "SUCCESS", "data": {"frame_quality": 4, "exercise_quality": "good",
                                                                   "exercise_performance": 48,
                                                                   "message": "Exercise successfully scored"}}

        # globals.results[video_id] = {"status": "FAIL", "message": "someerror"}

        # globals.results[video_id] = {"status": "SUCCESS", "data": {"frame_quality": 4, "exercise_quality": "bad",
        #                                                            "exercise_performance": None,
        #                                                            "message": "Bad exercise quality."}}

        # globals.results[video_id] = {"status": "SUCCESS", "data": {"frame_quality": 2, "exercise_quality": None,
        #                                                            "exercise_performance": None,
        #                                                            "message": "Bad recording quality"}}

        globals.progress.update({video_id : 100})
        video_writer.release()
        cap.release()
