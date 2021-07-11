import tensorflow as tf
import pandas as pd
import cv2
import time
import argparse
import sys
import posenet
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--video_width', type=int, default=1280)
parser.add_argument('--video_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--input_file', type=str, default=None)
parser.add_argument('--input_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='.\output')
args = parser.parse_args()
columns = ['Frame', 'Pose', 'Pose_Score', 'Nose_score', 'Nose_X_Coord', 'Nose_Y_Coord',
           'LeftEye_score', 'LeftEye_X_Coord', 'LeftEye_Y_Coord', 'RightEye_score', 'RightEye_X_Coord',
           'RightEye_Y_Coord', 'LeftEar_score', 'LeftEar_X_Coord', 'LeftEar_Y_Coord',
           'RightEar_score', 'RightEar_X_Coord', 'RightEar_Y_Coord', 'LeftShoulder_score', 'LeftShoulder_X_Coord',
           'LeftShoulder_Y_Coord', 'RightShoulder_score', 'RightShoulder_X_Coord', 'RightShoulder_Y_Coord',
           'LeftElbow_score', 'LeftElbow_X_Coord', 'LeftElbow_Y_Coord', 'RightElbow_score', 'RightElbow_X_Coord',
           'RightElbow_Y_Coord', 'LeftWrist_score', 'LeftWrist_X_Coord', 'LeftWrist_Y_Coord',
           'RightWrist_score', 'RightWrist_X_Coord', 'RightWrist_Y_Coord', 'LeftHip_score', 'LeftHip_X_Coord',
           'LeftHip_Y_Coord', 'RightHip_score', 'RightHip_X_Coord', 'RightHip_Y_Coord', 'LeftKnee_score', 'LeftKnee_X_Coord',
           'LeftKnee_Y_Coord', 'RightKnee_score', 'RightKnee_X_Coord', 'RightKnee_Y_Coord', 'LeftAnkle_score', 'LeftAnkle_X_Coord',
           'LeftAnkle_Y_Coord', 'RightAnkle_score', 'RightAnkle_X_Coord', 'RightAnkle_Y_Coord']
sess = tf.Session()

def main():
    model_cfg, model_outputs = posenet.load_model(args.model, sess)
    output_stride = model_cfg['output_stride']
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    if (args.input_dir is not None and not os.path.exists(args.input_dir)):
        sys.exit('Please make sure input directory is an existing directory')
    if args.input_dir is not None:
        filenames = [f.path for f in os.scandir(args.input_dir) if f.is_file()]
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                process_image(f, model_outputs, output_stride)
            elif f.lower().endswith(('.mp4')):
                process_video(f, model_outputs, output_stride)
                pass
            else:
                print('%s not supported, skipping' %f)
    elif args.input_file is not None:
        if args.input_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(args.input_file, model_outputs, output_stride)
            pass
        elif args.input_file.lower().endswith(('.mp4')):
            process_video(args.input_file, model_outputs, output_stride)
            pass
    else:
        sys.exit('Please specify input file or directory in arguments')
    sess.close()
    print("Finished!")


def process_image(file, model_outputs, output_stride):
    input_image, draw_image, output_scale = posenet.read_imgfile(
        file, scale_factor=args.scale_factor, output_stride=output_stride)

    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
        model_outputs,
        feed_dict={'image:0': input_image}
    )
    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
        heatmaps_result.squeeze(axis=0),
        offsets_result.squeeze(axis=0),
        displacement_fwd_result.squeeze(axis=0),
        displacement_bwd_result.squeeze(axis=0),
        output_stride=output_stride,
        max_pose_detections=10,
        min_pose_score=0.25)

    keypoint_coords *= output_scale

    if args.output_dir:
        draw_image = posenet.draw_skel_and_kp(
            draw_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.25, min_part_score=0.25)
        cv2.imwrite(os.path.join(args.output_dir,
                                 os.path.basename(file)), draw_image)
        print('Image created: ', os.path.join(
            args.output_dir, os.path.basename(file)))
        poseArray = []
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            # print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            tmp = [0, pi, pose_scores[pi]]  # pose num, pose score
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                #print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                tmp.append(s)  # part score
                tmp.append(c[0])  # part X coord
                tmp.append(c[1])  # part Y coord
            poseArray.append(tmp)
        create_csv(poseArray, file)



def process_video(file, model_outputs, output_stride):
     cap = cv2.VideoCapture(file)
     cap.set(3, args.video_width)
     cap.set(4, args.video_height)
     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
     out = cv2.VideoWriter(os.path.join(args.output_dir,
                           os.path.splitext(os.path.basename(file))[0]+'.mp4'),
                            fourcc, 20.0, (args.video_width, args.video_height))
     frame_count = 0
     poseArray = []
     while frame_count<int(cap.get(cv2.CAP_PROP_FRAME_COUNT)): 
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)
        keypoint_coords *= output_scale

        overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
        cv2.imshow('posenet', overlay_image)
        out.write(overlay_image)
        frame_count += 1
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            tmp = [frame_count, pi, pose_scores[pi]]  # pose num, pose score
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                tmp.append(s)  # part score
                tmp.append(c[0])  # part X coord
                tmp.append(c[1])  # part Y coord
            poseArray.append(tmp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
     cv2.destroyAllWindows()
     cap.release()
     out.release()
     print('Video Created: ', os.path.join(
            args.output_dir, os.path.basename(file)))
     create_csv(poseArray, file)


def create_csv(poseArray, file):
    df = pd.DataFrame(data=poseArray, columns=columns)
    df.to_csv(os.path.join(args.output_dir,
                           os.path.splitext(os.path.basename(file))[0]+'.csv'))
    print('CSV created: ', os.path.join(args.output_dir,
                                        os.path.splitext(os.path.basename(file))[0]+'.csv'))


if __name__ == "__main__":
    main()
