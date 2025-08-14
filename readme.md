🏏 AthleteRise: AI-Powered Cricket Cover Drive Analyzer
An AI-powered web application built with MediaPipe to analyze the biomechanics of a cricket cover drive from a video. The tool provides quantitative scores and actionable feedback on key aspects of the batting technique.

✨ Features

AI Pose Estimation: Uses Google's MediaPipe to detect 33 body landmarks in each frame.

Biomechanical Analysis: Calculates critical metrics for a successful cover drive, including:

Front Elbow Extension: Measures the straightness of the front arm at impact for power and control.

Head-to-Foot Alignment: Measures balance by checking if the head is positioned correctly over the front foot.

Performance Scoring: Provides a score out of 10 for each key metric.

Annotated Video Output: Generates a new video with the pose skeleton overlaid, showing the analysis in action.

Downloadable Reports: Allows you to download both the annotated video and a JSON file with the detailed scores.

🛠️ Setup and Installation
To run this project locally, follow these steps:

1. Clone the Repository

git clone https://github.com/parthCJ/cricker-cover-drive-analyzer.git
cd cricker-cover-drive-analyzer

2. Create a Virtual Environment
It's recommended

# For Windows
python -m venv .venv
.\.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt


4. Download the Pose Landmarker Model

Download: pose_landmarker_heavy.task


💻 Technology Stack
Backend: Python

Computer Vision: OpenCV

Pose Estimation: Google MediaPipe

Numerical Operations: NumPy

Research-Papers => https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
                => https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
                => https://www.pythoncentral.io/yt-dlp-download-youtube-videos/
                => https://www.researchgate.net/publication/366665948_Pose_Recognition_in_Cricket_using_Keypoints
                => https://www.researchgate.net/publication/7929592_Off-side_front_foot_drives_in_men's_high_performance_cricket_an_in-situ_3D_biomechanical_analysis