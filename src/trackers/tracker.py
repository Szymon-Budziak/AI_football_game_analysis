import os
import pickle
import sys
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

__all__ = ["Tracker"]

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()  # back fill to fill first missing values, if there are any

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames: list) -> list:
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # we want to track not predict
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames: list, read_from_stub=False, stub_path=None) -> dict:
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Dictionary to track class names
        tracks = {"players": [],  # {0: {'bbox': [0, 0, 0, 0], 1: {'bbox': [0, 0, 0, 0], 21:...} - this is for frame 1
                  "referees": [],
                  "ball": []
                  }

        # Get detections first
        detections = self.detect_frames(frames)

        # Override goalkeeper with player label because it switches between the two
        # Loop over detections
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # {0:player, 1:goal, ...}
            cls_names_inv = {v: k for k, v in cls_names.items()}  # {player:0, goal:1, ...}, switch key, values

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()  # bbox is 0 index in frame_detection
                cls_id = frame_detection[3]  # class id is 3 index in frame_detection
                track_id = frame_detection[4]  # track id is 4 index in frame_detection

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw ellipse under the player/referee
        cv2.ellipse(frame, center=(x_center, y2), axes=(width, int(0.35 * width)), angle=0.0, startAngle=-45,
                    endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)

        # Draw rectangle with number of the player
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

            # Draw number on the rectangle
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(frame, f'{track_id}', (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 0, 0), thickness=2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x_center, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([[x_center, y],
                                    [x_center - 10, y - 20],
                                    [x_center + 10, y - 20]], dtype=np.int32)

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)  # or - 1 for FILLED
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)  # draw just the black contour

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # semi-transparent rectangle

        # Calculate percentage that team has the ball
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # Make statistics
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames) * 100
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames) * 100

        cv2.putText(frame, f'Team 1 Ball Control: {team_1:.2f}%', (1400, 900), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 3)
        cv2.putText(frame, f'Team 2 Ball Control: {team_2:.2f}%', (1400, 950), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames: list, tracks: dict, team_ball_control) -> list:
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # copy to not override the original frame

            player_dict = tracks['players'][frame_num]
            referees_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)  # color BGR

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 0, 255))

            # Draw referee
            for track_id, referee in referees_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))

            # Draw team ball control statistic
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

    def add_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if obj == 'ball':
                        position = get_center_of_bbox(bbox)  # center of ball
                    else:
                        position = get_foot_position(bbox)  # foot position of player/referee

                    tracks[obj][frame_num][track_id]['position'] = position
