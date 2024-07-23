from trackers import Tracker
from utils import read_video, save_video


def main():
    # Read video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initialize tracker and run prediction on it
    tracker = Tracker("model/best.pt")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="output_videos/stub.txt")

    # Save video
    save_video(video_frames, "output_videos/output.avi")


if __name__ == "__main__":
    main()
