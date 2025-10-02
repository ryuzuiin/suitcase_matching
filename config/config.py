# config/config.py
import os

# === Paths (Colab) ===
# BASE_DIR = "/content/drive/MyDrive/南海電鉄"
BASE_DIR = "/content/drive/Shareddrives/T82_18_9001_一時保管2週間/劉→吉井"
INPUT_PATH = os.path.join(BASE_DIR, "data/JR高崎駅_新幹線改札口.mp4")

# === Models ===
POSE_MODEL_NAME = "yolo11m-pose"   # or "yolo11m-pose.pt"
SUITCASE_MODEL_NAME = "yolo11x"    # or "yolo11x.pt"

# === Video I/O ===
TARGET_FPS = 10
TEST_DURATION_SECONDS = 5

# === Outputs ===
# OUTPUT_DIR = os.path.join(BASE_DIR, f"{POSE_MODEL_NAME}_output")
OUTPUT_DIR = os.path.join(BASE_DIR, f"output")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, f"output_{POSE_MODEL_NAME}_{TARGET_FPS}fps.mp4")

# === Aliases for main import compatibility ===
output_dir = OUTPUT_DIR
input_path = INPUT_PATH
output_video_path = OUTPUT_VIDEO_PATH
target_fps = TARGET_FPS
model_name = POSE_MODEL_NAME
test_duration_seconds = TEST_DURATION_SECONDS
suitcase_model_name = SUITCASE_MODEL_NAME
