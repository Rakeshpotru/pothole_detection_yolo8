import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
from base64 import b64encode
import tempfile

def save_uploaded_file(uploaded_file):
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def calculate_focal_length(pixel_length, distance_to_object, image_width_pixels):
    # Using the formula: F = (P * D) / W
    focal_length = (pixel_length * distance_to_object) / image_width_pixels
    return focal_length

def calculate_real_world_length(focal_length, pixel_length, image_width_pixels, DISTANCE_TO_OBJECT):
    # Using the formula: W = (P * F) / D
    real_world_length = (pixel_length * focal_length) / DISTANCE_TO_OBJECT
    return real_world_length

def draw_detections(image, detections,shape):
    draw = ImageDraw.Draw(image)
    height, width =shape
    # with Image.open(image) as img:
    #     shape = np.array(img).shape

    font = ImageFont.load_default()
    
    # Conversion factor from pixels to meters
    pixel_to_meter = 0.001  # Update this based on your actual measurement
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf, cls_id = det[4:]
        #pixel_to_cm = 0.026458
        # Constants
        DISTANCE_TO_OBJECT = 100  # Distance from camera to object in cm (0.9 meters)
        IMAGE_WIDTH_PIXELS = width  # Example image width in pixels
        FOCAL_LENGTH = None  # Focal length (to be calculated)  
        pixel_length = y2 - y1  # Length of the object in pixels
        # Calculate focal length based on the provided values
        FOCAL_LENGTH = calculate_focal_length(pixel_length, DISTANCE_TO_OBJECT, IMAGE_WIDTH_PIXELS)

        # Calculate the real-world dimensions
        real_world_length = calculate_real_world_length(FOCAL_LENGTH, pixel_length, IMAGE_WIDTH_PIXELS,DISTANCE_TO_OBJECT)
        real_world_width = calculate_real_world_length(FOCAL_LENGTH, x2 - x1, IMAGE_WIDTH_PIXELS, DISTANCE_TO_OBJECT)
        # Calculate area in meters²
        # area_pixels = (x2 - x1) * (y2 - y1)
        # area_m2 = area_pixels * (pixel_to_meter ** 2)
        area_cm2 = real_world_length * real_world_width
        area_m2 = area_cm2 /10000


        label = f'{int(cls_id)} {conf:.2f} | Area: {area_m2:.2f} m²'
        st.success(f"Area {area_m2:.2f} m² ")
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        text_width = font.getlength(label)
        text_height = font.getbbox(label)[3]  # Get the height from the bounding box
        draw.rectangle([x1, y1, x1 + text_width, y1 + text_height + 2], fill=(255, 0, 0), outline=(255, 0, 0))
        draw.text((x1, y1), label, font=font, fill=(255, 255, 255))
    
    return image

# def draw_detections(image, detections):
#     draw = ImageDraw.Draw(image)
#     font = ImageFont.truetype("arial.ttf", 16)  # Replace with your desired font and size
#     for det in detections:
#         x1, y1, x2, y2 = map(int, det[:4])
#         conf, cls_id = det[4:]
#         label = f'{int(cls_id)} {conf:.2f}'
#         draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
#         text_width = font.getlength(label)
#         text_height = font.getbbox(label)[3]  # Get the height from the bounding box
#         draw.rectangle([x1, y1, x1 + text_width, y1 + text_height + 2], fill=(255, 0, 0), outline=(255, 0, 0))
#         draw.text((x1, y1), label, font=font, fill=(255, 255, 255))
#     return image

def generate_roadmap(image, detections, threshold=5):
    road_blocked = len(detections) > threshold
    if road_blocked:
        draw = ImageDraw.Draw(image)
        draw.line([(0, 0), (image.width, image.height)], fill=(255, 0, 0), width=10)
        #draw.text((10, 10), "Road Blocked", font=ImageFont.truetype("arial.ttf", 36), fill=(255, 0, 0))
    return image, road_blocked

st.title('Pothole Detection')
current_dir = os.path.dirname(__file__)
# Model selection
model_version = st.selectbox('Choose your model:', ('yolov8n', 'yolov8m'))
model_paths = {
    'yolov8n': current_dir + '//content//runs//detect//train5//weights//best.pt',
    'yolov8m': current_dir + '//content//runs//detect//train5//weights//best.pt'
}
model_path = model_paths[model_version]
model = YOLO(model_path)

uploaded_file = st.file_uploader("Upload an image or video...", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)

    if uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
        results = model(file_path, conf=0.2)

        # Iterate over the results and display each processed image
        for result in results:
            result_image = result.orig_img
            shape = 1280,720
            st.image(result_image, caption='Processed Image', use_column_width=True)

            detections = result.boxes.data.tolist()
            if detections:
                annotated_image = Image.fromarray(result_image)
                annotated_image = draw_detections(annotated_image, detections, shape)
                roadmap_image, road_blocked = generate_roadmap(annotated_image, detections, threshold=5)
                st.image(roadmap_image, caption='Roadmap', use_column_width=True)
                if road_blocked:
                    st.warning("Too many potholes detected. Road is blocked for cars.")
                else:
                    st.success("Road is clear for vehicles to drive through.")
            else:
                st.success("No Detections identified for the above media ")

    elif uploaded_file.name.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            st.error("Failed to open the video file.")
        else:
            st.write("Processing video...")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)

            # Create a temporary directory for storing processed frames
            with tempfile.TemporaryDirectory() as temp_dir:
                processed_frames = []

                for frame_idx in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=0.1)

                    for result in results:
                        detections = result.boxes.data.tolist()
                        if detections:
                            result_image = result.orig_img
                            annotated_image = Image.fromarray(result_image)
                            annotated_image = draw_detections(annotated_image, detections)
                            roadmap_image, road_blocked = generate_roadmap(annotated_image, detections, threshold=5)
                            st.image(roadmap_image, caption='Processed Frame', use_column_width=True)
                            if road_blocked:
                                st.warning("Too many potholes detected. Road is blocked for cars.")

                            # Save the processed frame to the temporary directory
                            frame_path = os.path.join(temp_dir, f"frame_{frame_idx}.jpg")
                            roadmap_image.save(frame_path)
                            processed_frames.append(frame_path)

                    progress_bar.progress((frame_idx + 1) / frame_count)

                cap.release()

                # Combine the processed frames into a video
                output_video_path = os.path.join(temp_dir, "output_video.mp4")
                frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (int(frame_width), int(frame_height)))

                for frame_path in processed_frames:
                    frame = cv2.imread(frame_path)
                    video_writer.write(frame)

                video_writer.release()

                # Display the combined video
                video_bytes = open(output_video_path, "rb").read()
                st.video(video_bytes)

    # Cleanup: Remove the uploaded file to clear space
    os.remove(file_path)