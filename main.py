import cv2
import mediapipe as mp
import numpy as np
import yt_dlp
import json
import os
import time
import math

# --- 2. Configuration ---

# this section set all the parameters
CONFIG = {

    "video_url": "https://youtube.com/shorts/vSX3IRxGnNY",
    "model_path": "pose-landmark-model\pose_landmarker_heavy (1).task",
    "data_paths": {
        "input": "data/input",
        "output": "data/output",
        "source_video": "data/input/source_video.mp4",
        "annotated_video": "data/output/annotated_video.mp4",
        "evaluation_report": "data/output/evaluation.json",
    },
    # For a right-handed batsman. 
    "keypoint_mapping": {
        "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28, "left_heel": 29, "right_heel": 30,
        "left_foot_index": 31, "right_foot_index": 32,
    },
    "analysis_thresholds": {
        "visibility_min": 0.6,
        "good_elbow_angle": 160.0,  # extension angle for a good cover drive
        "bad_head_alignment": 0.15, # Normalized distance
    },
    "output_video": {
        "codec": "XVID",  # Better codec for compatibility
        "width": 1920,   
        "height": 1080,  
        "maintain_aspect_ratio": True,
    }
}

# --- 3.Biomechanical Calculation Functions ---

def download_video(url, output_path):
    """Downloads a video from a URL using yt-dlp if it doesn't already exist."""
    if os.path.exists(output_path):
        print(f"Video already exists at {output_path}. Skipping download.")
        return
    print(f"Downloading video from {url} to {output_path}...")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best',
        'outtmpl': output_path,
        'quiet': False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading video: {e}")
        exit() # Exit if download fails

def calculate_angle_3d(a, b, c):
    """Calculates the angle between three 3D points (e.g., shoulder, elbow, wrist)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    vec_ba = a - b
    vec_bc = c - b
    dot_product = np.dot(vec_ba, vec_bc)
    norm_product = np.linalg.norm(vec_ba) * np.linalg.norm(vec_bc) # normalising the vector.
    if norm_product == 0:
        return 0.0 # Avoid division by zero
        # np.clip for avoiding the minor floating point errors
    cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0) # .clip for setting the angle between -1.0 and 1.0
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def get_landmark_coords(landmarks, keypoint_index, dims=3):
    """Safely retrieves landmark coordinates."""
    lm = landmarks[keypoint_index]
    if dims == 3:
        return [lm.x, lm.y, lm.z]
    return [lm.x, lm.y]

def create_landmark_list_from_landmarks(landmarks):
    """Convert landmarks list to NormalizedLandmarkList for drawing."""
    from mediapipe.framework.formats import landmark_pb2
    
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for landmark in landmarks:
        landmark_proto = landmark_list.landmark.add()
        landmark_proto.x = landmark.x
        landmark_proto.y = landmark.y
        landmark_proto.z = landmark.z
        landmark_proto.visibility = getattr(landmark, 'visibility', 1.0)
        landmark_proto.presence = getattr(landmark, 'presence', 1.0)
    
    return landmark_list

# --- 4. Core Analysis Pipeline ---

def analyze_cover_drive():
    """Main function to run the full analysis pipeline."""
    print("🚀 Starting AthleteRise analysis...")
    
    # --- Setup Phase ---
    os.makedirs(CONFIG["data_paths"]["input"], exist_ok=True)
    os.makedirs(CONFIG["data_paths"]["output"], exist_ok=True)
    download_video(CONFIG["video_url"], CONFIG["data_paths"]["source_video"])

    # Check if the video file exists and is readable
    if not os.path.exists(CONFIG["data_paths"]["source_video"]):
        print("❌ Error: Source video file not found!")
        return
    
    # Test video file integrity
    test_cap = cv2.VideoCapture(CONFIG["data_paths"]["source_video"])
    if not test_cap.isOpened():
        print("❌ Error: Cannot open source video file!")
        return
    test_cap.release()
    print("✅ Video file validated successfully")

    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    try:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=CONFIG["model_path"]),
            running_mode=VisionRunningMode.IMAGE
        )
        print("✅ MediaPipe model loaded successfully")
    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at '{CONFIG['model_path']}'.")
        print("Please download the 'pose_landmarker_heavy.task' file and place it in the correct directory.")
        return
    except Exception as e:
        print(f"❌ ERROR: Failed to load MediaPipe model: {e}")
        return
        
    # --- Processing Phase ---
    cap = cv2.VideoCapture(CONFIG["data_paths"]["source_video"])
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Original video: {original_width}x{original_height} @ {original_fps:.2f} FPS")
    
    # Calculate output dimensions maintaining aspect ratio
    target_width = CONFIG["output_video"]["width"]
    target_height = CONFIG["output_video"]["height"]
    
    if CONFIG["output_video"]["maintain_aspect_ratio"]:
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        # Adjust dimensions to maintain aspect ratio
        if aspect_ratio > (target_width / target_height):
            # Video is wider - fit to width
            output_width = target_width
            output_height = int(target_width / aspect_ratio)
        else:
            # Video is taller - fit to height
            output_height = target_height
            output_width = int(target_height * aspect_ratio)
        
        # Ensure dimensions are even.
        output_width = output_width if output_width % 2 == 0 else output_width - 1
        output_height = output_height if output_height % 2 == 0 else output_height - 1
    else:
        output_width, output_height = target_width, target_height
    
    print(f"📤 Output video: {output_width}x{output_height}")

    # Initialize video writer with better settings
    fourcc = cv2.VideoWriter_fourcc(*CONFIG["output_video"]["codec"])
    out = cv2.VideoWriter(
        CONFIG["data_paths"]["annotated_video"],
        fourcc, 
        original_fps,  # Use original FPS
        (output_width, output_height)
    )

    frame_count = 0
    all_metrics = []
    start_time = time.time()
    
    # Use 'with' block for automatic resource management
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert frame for MediaPipe
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Perform pose detection on the current frame
            
            detection_result = landmarker.detect(mp_image)

            frame_metrics = {"frame": frame_count}
            feedback_text = []

            if detection_result.pose_landmarks:
                # Access the landmarks for the first detected person ([0])
                landmarks = detection_result.pose_landmarks[0]
                k = CONFIG["keypoint_mapping"]
                vis_min = CONFIG["analysis_thresholds"]["visibility_min"]

                # *** FIX ***: Convert landmarks to proper format for drawing
                try:
                    # Convert landmarks list to NormalizedLandmarkList for drawing
                    landmark_list = create_landmark_list_from_landmarks(landmarks)
                    
                    # Draw the landmarks on the BGR frame for output video
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame_resized,
                        landmark_list,  # Pass the converted landmark list
                        mp.solutions.pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                    )
                except Exception as e:
                    print(f"Warning: Could not draw landmarks: {e}")
                    # Continue processing even if drawing fails

                # --- Metric Calculation ---
                
                # 1. Front Elbow Angle (assuming right-handed batsman)
                try:
                    if (len(landmarks) > k["left_shoulder"] and 
                        len(landmarks) > k["left_elbow"] and 
                        len(landmarks) > k["left_wrist"] and
                        landmarks[k["left_shoulder"]].visibility > vis_min and
                        landmarks[k["left_elbow"]].visibility > vis_min and
                        landmarks[k["left_wrist"]].visibility > vis_min):
                        
                        shoulder = get_landmark_coords(landmarks, k["left_shoulder"])
                        elbow = get_landmark_coords(landmarks, k["left_elbow"])
                        wrist = get_landmark_coords(landmarks, k["left_wrist"])
                        
                        elbow_angle = calculate_angle_3d(shoulder, elbow, wrist)
                        frame_metrics['elbow_angle'] = elbow_angle
                        cv2.putText(frame_resized, f"Elbow Angle: {elbow_angle:.1f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        if elbow_angle > CONFIG["analysis_thresholds"]["good_elbow_angle"]:
                            feedback_text.append("Good Arm Extension")
                except Exception as e:
                    print(f"Warning: Could not calculate elbow angle: {e}")
                    pass

                # 2. Head-over-Foot Alignment
                try:
                    if (len(landmarks) > k["nose"] and 
                        len(landmarks) > k["left_foot_index"] and
                        len(landmarks) > k["left_shoulder"] and
                        len(landmarks) > k["left_ankle"] and
                        landmarks[k["nose"]].visibility > vis_min and
                        landmarks[k["left_foot_index"]].visibility > vis_min):
                        
                        nose_x = landmarks[k["nose"]].x
                        foot_x = landmarks[k["left_foot_index"]].x
                        
                        # Use shoulder-ankle distance as a proxy for player height in the frame
                        shoulder_y = landmarks[k["left_shoulder"]].y
                        ankle_y = landmarks[k["left_ankle"]].y
                        player_height_proxy = abs(ankle_y - shoulder_y)
                        
                        head_alignment_offset = abs(nose_x - foot_x)
                        # Normalize by height to make it scale-invariant
                        head_alignment_norm = head_alignment_offset / player_height_proxy if player_height_proxy > 0 else 0
                        
                        frame_metrics['head_alignment'] = head_alignment_norm
                        cv2.putText(frame_resized, f"Head Align: {head_alignment_norm:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        if head_alignment_norm > CONFIG["analysis_thresholds"]["bad_head_alignment"]:
                            feedback_text.append("Head Not Over Foot")
                except Exception as e:
                    print(f"Warning: Could not calculate head alignment: {e}")
                    pass

            # Display feedback cues on the frame
            for i, text in enumerate(feedback_text):
                cv2.putText(frame_resized, text, (50, 200 + i*50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            out.write(frame_resized)
            frame_count += 1
            if len(frame_metrics) > 1: # Only add if metrics were calculated
                all_metrics.append(frame_metrics)

            # Progress indicator
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Processed {frame_count} frames...")

    # --- Reporting Phase ---
    end_time = time.time()
    processing_time = end_time - start_time
    avg_fps = frame_count / processing_time if processing_time > 0 else 0
    print(f"\n✅ Processing finished. Analyzed {frame_count} frames.")
    print(f"⏱️ Average processing speed: {avg_fps:.2f} FPS.")

    # Generate final evaluation
    evaluation = {
        "performance_summary": {},
        "actionable_feedback": {},
        "processing_stats": {
            "total_frames": frame_count,
            "avg_fps": round(avg_fps, 2),
            "metrics_frames": len(all_metrics)
        }
    }

    # *** CORRECTION ***: Fixed the report generation logic to build the dictionary correctly.
    if all_metrics:
        # Elbow Extension Score
        elbow_angles = [m.get('elbow_angle', 0) for m in all_metrics if 'elbow_angle' in m]
        if elbow_angles:
            max_elbow = max(elbow_angles)
            avg_elbow = np.mean(elbow_angles)
            elbow_score = np.clip((max_elbow - 140) / (180 - 140) * 10, 1, 10) # Scale 140-180 deg to 1-10 score
            evaluation["performance_summary"]["Elbow Extension"] = round(elbow_score, 1)
            evaluation["actionable_feedback"]["Elbow Extension"] = "Excellent arm extension, leading to a powerful shot." if elbow_score > 7 else "Focus on fully extending your front arm towards the ball at impact."
            print(f"📊 Elbow Analysis: Max={max_elbow:.1f}°, Avg={avg_elbow:.1f}°, Score={elbow_score:.1f}/10")

        # Head Position Score
        head_alignments = [m.get('head_alignment', 1) for m in all_metrics if 'head_alignment' in m]
        if head_alignments:
            min_head_align = min(head_alignments)
            avg_head_align = np.mean(head_alignments)
            head_score = np.clip((CONFIG["analysis_thresholds"]["bad_head_alignment"] - min_head_align) * 100, 1, 10)
            evaluation["performance_summary"]["Head Position"] = round(head_score, 1)
            evaluation["actionable_feedback"]["Head Position"] = "Great head position, staying stable and over the front foot." if head_score > 7 else "Work on keeping your head still and leaning into the shot to maintain balance."
            print(f"📊 Head Analysis: Min={min_head_align:.3f}, Avg={avg_head_align:.3f}, Score={head_score:.1f}/10")
    
    else:
        print("⚠️ Warning: No metrics were calculated. Check if pose detection is working properly.")
        evaluation["performance_summary"]["Status"] = "No valid poses detected"
        evaluation["actionable_feedback"]["Status"] = "Unable to analyze technique - ensure the video shows a clear view of the batsman."

    # Save the evaluation report
    report_path = CONFIG["data_paths"]["evaluation_report"]
    with open(report_path, 'w') as f:
        json.dump(evaluation, f, indent=4)
    
    print(f"📊 Evaluation report saved to {report_path}")

    # --- Cleanup ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Verify output video was created successfully
    if os.path.exists(CONFIG["data_paths"]["annotated_video"]):
        output_size = os.path.getsize(CONFIG["data_paths"]["annotated_video"])
        if output_size > 0:
            print(f"✅ Output video created successfully: {output_size / (1024*1024):.2f} MB")
        else:
            print("⚠️ Warning: Output video file is empty!")
    else:
        print("❌ Error: Output video file was not created!")
    
    print("🎬 Analysis complete. Outputs are in the 'data/output' directory.")


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    analyze_cover_drive()