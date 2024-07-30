# AI Football Game Analysis

This project aims to **analyze real-time football**, detect and track players, referees, and ball in a video using
**YOLO**, a state-of-the-art AI object detection model. The model's performance was enhanced through custom training.
Players are assigned to teams based on their t-shirt colors using **K-means** for pixel segmentation and clustering.
This allows to **measure a team's ball acquisition** percentage during a match.

**Optical flow** was employed to measure camera movement between frames, enabling precise measurement of player
movement. **Perspective transformation** was implemented to represent the scene's depth and perspective, allowing us to
measure player movement in meters rather than pixels.

Finally, player speed and the distance covered was calculated to provide in depth statistics.

# Installation

The projects uses Poetry to manage dependencies. All the dependencies are in `pyproject.toml`. To install the them, run
the following command:

```bash
poetry install
```

# Output video

A screenshots from the output video:

![image_result_1](src/data/image_result_1.png)

![image_result_2](src/data/image_result_2.png)

# Models used

- **YOLO v8** for player, referee, and ball detection, YOLO performance was enhanced through custom training
- **Kmeans** for pixel segmentation and clustering to detect t-shirt color
- **Optical Flow** to measure camera movement
- **Perspective Transformation** to represent scene depth and perspective
- **Speed and distance calculation** per player

# Training

Training for the players, referees, and ball detection is written in notebook:

- `src/training/football_training_yolo.ipynb`
