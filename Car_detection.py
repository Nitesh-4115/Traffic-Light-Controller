import streamlit as st
import tempfile
import cv2
import os
import supervision as sv  # Make sure to install the Supervision library
import sys
import subprocess
#!{sys.executable} -m pip install huggingface_hub
from ultralytics import YOLOv10
from moviepy.video.io.VideoFileClip import VideoFileClip

def reencode_video(input_path, output_path):
    video = VideoFileClip(input_path)
    video.write_videofile(output_path, codec="libx264")

model = YOLOv10('best.pt')

def process_video(input_path, output_path):
    """
    Processes the video by annotating bounding boxes and labels using your logic.
    """
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process each frame
        results = model(source=frame, conf=0.25)[0]
        detections = sv.Detections.from_ultralytics(results)
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        
        out.write(annotated_frame)
    
    cap.release()
    out.release()

# Streamlit UI
st.title("Traffic Police Video Annotation App")
st.write("Upload a video to annotate with bounding boxes and labels using YOLO.")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input_file.write(uploaded_video.read())
    temp_input_file.close()

    st.video(temp_input_file.name)  # Display the uploaded video

    # Process the video
    st.write("Processing video... This may take some time.")
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    process_video(temp_input_file.name, temp_output_file.name)

    # Re-encode the video for compatibility
    st.write("Re-encoding video to ensure compatibility...")
    reencoded_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    reencode_video(temp_output_file.name, reencoded_output_file.name)

    st.write("Video processing complete! Here's the output:")
    with open(reencoded_output_file.name, "rb") as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)  # Display the re-encoded video

        # Provide a download link for the processed video
        st.download_button(
            label="Download Annotated Video",
            data=video_bytes,
            file_name="annotated_video.mp4",
            mime="video/mp4"
        )

    # Register cleanup for when the session ends
    import atexit

    def cleanup():
        try:
            if os.path.exists(temp_input_file.name):
                os.remove(temp_input_file.name)
            if os.path.exists(temp_output_file.name):
                os.remove(temp_output_file.name)
            if os.path.exists(reencoded_output_file.name):
                os.remove(reencoded_output_file.name)
        except Exception as e:
            print(f"Error during cleanup: {e}")

    atexit.register(cleanup)