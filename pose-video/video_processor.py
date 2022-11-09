from copy import deepcopy
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from keras.backend import set_session
import pandas as pd
from pathlib import Path
import globals

import posenet


class VideoProcessor:
    def __init__(self, pose_net_estimator, pose2kinect_estimator, kinect3d_estimator, key_frame_estimator,
                 frame_quality_estimator, exercise_quality_recognizer, exercise_performance_scorer):
        self._pose_net_estimator = pose_net_estimator
        self._pose2kinect_estimator = pose2kinect_estimator
        self._kinect3d_estimator = kinect3d_estimator
        self._key_frame_estimator = key_frame_estimator
        self._frame_quality_estimator = frame_quality_estimator
        self._exercise_quality_recognizer = exercise_quality_recognizer
        self._exercise_performance_scorer = exercise_performance_scorer
        self._eval_freq = 5
        self._frame_confid_thres = 0.2
        self._frame_score_thres = 1

        self._success_status = "SUCCESS"
        self._fail_status = "FAIL"
        self._response_template = {"status": f"{self._success_status}",
                                   "data": {"frame_quality": "None",
                                            "exercise_quality": "None",
                                            "exercise_performance": "None",
                                            "message": "Exercise successfully scored"}}

    def process(self, video_path, csv_path, session, graph, video_id):
        try:
            response = self.process_response(video_path, csv_path, session, graph, video_id)
        except Exception as e:
            response = {"status": f"{self._fail_status}", "message": f"Error during processing: {e}"}
        globals.results[video_id] = response
        globals.progress.update({video_id: 100})

    def process_response(self, video_path, csv_path, session, graph, video_id):
        input_video = NamedTemporaryFile(suffix=Path(video_path).suffix)
        with open(video_path, "rb") as from_v:
            data = from_v.read()

        with open(input_video.name, "wb") as to_v:
            to_v.write(data)

        cap = cv2.VideoCapture(input_video.name)
        frame_pause = int(cap.get(cv2.CAP_PROP_FPS) / self._eval_freq)

        kinect3d_df, all_pose_metrics, start_frame_i, stop_frame_i = self._get_kinect3d_analysis(
            cap, video_id, frame_pause, graph, session)
        kinect3d_df.to_csv(csv_path, index=False)
        self._write_processed_video(cap, video_path, video_id, all_pose_metrics, start_frame_i, stop_frame_i)

        response = deepcopy(self._response_template)
        score, confidence = self._get_frame_quality(kinect3d_df, cap, start_frame_i, graph, session)
        response["data"]["frame_quality"] = score
        if score > self._frame_score_thres and confidence > self._frame_confid_thres:
            exercise_quality = self._get_exercise_quality(kinect3d_df, start_frame_i, stop_frame_i, graph, session)
            response["data"]["exercise_quality"] = "Good" if exercise_quality == 1 else "Bad"
            if exercise_quality == 1:
                exercise_performance = self._get_exercise_performance(
                    kinect3d_df, start_frame_i, stop_frame_i, graph, session)
                response["data"]["exercise_performance"] = exercise_performance
            else:
                response["data"]["message"] = "Bad exercise quality"
        else:
            response["data"]["message"] = "Bad recording quality"
        cap.release()
        return response

    def _get_kinect3d_analysis(self, cap, video_id, frame_pause, graph, session):
        all_poses_df = pd.DataFrame(data={})
        all_pose_metrics = {}
        ret, frame = cap.read()

        while ret:
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_progress = int(frame_index / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 70)
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
        return kinect3d_df, all_pose_metrics, start_frame_i, stop_frame_i

    def _write_processed_video(self, cap, video_path, video_id, all_pose_metrics, start_frame_i, stop_frame_i):
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
            new_progress = int(((frame_index - start_frame_i) / (stop_frame_i - start_frame_i) * 20) + 70)
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

        globals.progress.update({video_id: 90})
        video_writer.release()

    def _get_frame_quality(self, kinect3d, cap, start_frame_i, graph, session):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_i)
        input_kinect3d = kinect3d.iloc[:self._frame_quality_estimator.frames_to_collect].values
        input_frames = []

        ret, frame = cap.read()

        while ret and len(input_frames) < self._frame_quality_estimator.frames_to_collect:
            input_frames.append(frame)
            ret, frame = cap.read()

        with graph.as_default():
            set_session(session)
            score, confidence = self._frame_quality_estimator.predict(np.array(input_frames), input_kinect3d)
        return score, confidence

    def _get_exercise_quality(self, kinect3d, start_frame_i, stop_frame_i, graph, session):
        frames_indexes = self._distribute_key_frames(
            start_frame_i, stop_frame_i, number_of_keys=self._exercise_quality_recognizer.number_of_frames)
        kinect_input = kinect3d.loc[kinect3d["FrameNum"].isin(frames_indexes)]
        kinect_input = kinect_input.drop(columns=["FrameNum"]).values
        with graph.as_default():
            set_session(session)
            quality = self._exercise_quality_recognizer.predict(kinect_input)
        quality = int(quality)
        return quality

    def _get_exercise_performance(self, kinect3d, start_frame_i, stop_frame_i, graph, session):
        frames_indexes = self._distribute_key_frames(
            start_frame_i, stop_frame_i, number_of_keys=self._exercise_performance_scorer.number_of_frames)
        kinect_input = kinect3d.loc[kinect3d["FrameNum"].isin(frames_indexes)]
        kinect_input = kinect_input.drop(columns=["FrameNum"]).values
        with graph.as_default():
            set_session(session)
            performance = self._exercise_performance_scorer.predict(kinect_input)
        performance = int(performance * 100)
        return performance

    def _distribute_key_frames(self, start_frame, end_frame, number_of_keys=10):
        distributed_keys = []
        step = (end_frame - start_frame) / (number_of_keys - 1)
        prev_key = start_frame
        distributed_keys.append(start_frame)

        for i in range(number_of_keys - 1):
            current_key = int(prev_key + step)
            distributed_keys.append(current_key)
            prev_key = current_key
        distributed_keys.append(end_frame)
        return distributed_keys
