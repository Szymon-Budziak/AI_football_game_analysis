import cv2


def read_video(video_path: str) -> list:
    frames = []

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    return frames


def save_video(output_video_frames: list, output_video_path: str) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        20.0,  # 20.0 is the frame rate (frames per second)
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )

    for frame in output_video_frames:
        out.write(frame)

    out.release()
