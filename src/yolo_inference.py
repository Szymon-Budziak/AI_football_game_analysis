from ultralytics import YOLO

if __name__ == "__main__":
    # Initialize model
    # model = YOLO("models/yolov8m.pt")
    model = YOLO("models/best.pt")

    # Inference
    results = model.predict("input_videos/08fd33_4.mp4", save=True)

    print(results[0])

    for box in results[0].boxes:
        print(box)
