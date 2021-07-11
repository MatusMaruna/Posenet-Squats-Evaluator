import numpy as np
import pandas as pd
import tensorflow as tf

import posenet
import posenet.constants
from posenet.utils import _process_input


class PosenetEstimator:

    def __init__(self, model, scale_factor=0.7125):
        self.sess = tf.Session()
        self.model = model
        self.model_cfg, self.model_outputs = posenet.load_model(self.model, self.sess)
        self.output_stride = self.model_cfg['output_stride']
        self.scale_factor = scale_factor
        self.columns = ['Frame', 'Pose', 'Pose_Score', 'Nose_score', 'Nose_X_Coord', 'Nose_Y_Coord',
                        'LeftEye_score', 'LeftEye_X_Coord', 'LeftEye_Y_Coord', 'RightEye_score', 'RightEye_X_Coord',
                        'RightEye_Y_Coord', 'LeftEar_score', 'LeftEar_X_Coord', 'LeftEar_Y_Coord',
                        'RightEar_score', 'RightEar_X_Coord', 'RightEar_Y_Coord', 'LeftShoulder_score',
                        'LeftShoulder_X_Coord',
                        'LeftShoulder_Y_Coord', 'RightShoulder_score', 'RightShoulder_X_Coord', 'RightShoulder_Y_Coord',
                        'LeftElbow_score', 'LeftElbow_X_Coord', 'LeftElbow_Y_Coord', 'RightElbow_score',
                        'RightElbow_X_Coord',
                        'RightElbow_Y_Coord', 'LeftWrist_score', 'LeftWrist_X_Coord', 'LeftWrist_Y_Coord',
                        'RightWrist_score', 'RightWrist_X_Coord', 'RightWrist_Y_Coord', 'LeftHip_score',
                        'LeftHip_X_Coord',
                        'LeftHip_Y_Coord', 'RightHip_score', 'RightHip_X_Coord', 'RightHip_Y_Coord', 'LeftKnee_score',
                        'LeftKnee_X_Coord',
                        'LeftKnee_Y_Coord', 'RightKnee_score', 'RightKnee_X_Coord', 'RightKnee_Y_Coord',
                        'LeftAnkle_score',
                        'LeftAnkle_X_Coord',
                        'LeftAnkle_Y_Coord', 'RightAnkle_score', 'RightAnkle_X_Coord', 'RightAnkle_Y_Coord']
        self.eval_columns_constrains = '^Unnamed|^Frame|^Pose|Eye|Ear|_z|_score|_Score'
        self.score_columns_constrains = 'Eye|Ear'

    def evaluate(self, image, return_raw_output=False):
        process_result = self.process_image(image, return_raw_output=return_raw_output)
        if return_raw_output:
            processed_df, raw_output = process_result
            output_df = processed_df.loc[:, ~processed_df.columns.str.contains(self.eval_columns_constrains)]
            output = output_df, raw_output
        else:
            output = process_result.loc[:, ~process_result.columns.str.contains(self.eval_columns_constrains)]
        return output

    def evaluate_with_confidence(self, image):
        process_result = self.process_image(image, return_raw_output=False)
        output = process_result.loc[:, ~process_result.columns.str.contains(self.score_columns_constrains)]
        return output

    def process_image(self, image, return_raw_output=False):

        input_image, draw_image, output_scale = _process_input(image, scale_factor=self.scale_factor,
                                                               output_stride=self.output_stride)

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = self.sess.run(
            self.model_outputs,
            feed_dict={'image:0': input_image}
        )

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=self.output_stride,
            max_pose_detections=10,
            min_pose_score=0.25)

        keypoint_coords *= output_scale
        poseArray = []
        pi = np.argmax(pose_scores)

        if pose_scores[pi] != 0.:

            pose_part = [0, pi, pose_scores[pi]]  # pose num, pose score
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                pose_part.append(s)  # part score
                pose_part.append(c[0])  # part X coord
                pose_part.append(c[1])  # part Y coord
            poseArray.append(pose_part)

        pose_result = pd.DataFrame(data=poseArray, columns=self.columns)
        if return_raw_output:
            pose_result = pose_result, (pose_scores, keypoint_scores, keypoint_coords)
        return pose_result
