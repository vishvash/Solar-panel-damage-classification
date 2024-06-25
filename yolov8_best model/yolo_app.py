import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model_path = r"C:\Users\ADMIN\Desktop\DS - Notes\Project1 - Solar Panel Damage Deduction\Project Initial Docs\models - yoloV8\best.pt"  # Replace with the actual path to your best.pt file
model = YOLO(model_path)  # Load your downloaded best.pt weights with YOLOv8

# Define function for object detection
def detect_objects(image):
    results = model(image)  # Perform inference with the YOLOv8 model
    detections = results[0].boxes  # Extract boxes object
    boxes = detections.xyxy.cpu().numpy()  # Extract bounding boxes
    labels = detections.cls.cpu().numpy().astype(int)  # Extract class labels as integers
    confidences = detections.conf.cpu().numpy()  # Extract confidence scores
    return boxes, labels, confidences

def draw_label(img, text, pos, bg_color):
    font_scale = 0.8
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
    text_x, text_y = pos
    box_coords = ((text_x, text_y), (text_x + text_size[0], text_y - text_size[1] - 10))
    cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(img, text, (text_x, text_y - 5), font, font_scale, (255, 255, 255), 2)

def main():
    """
    Streamlit application for object detection using YOLOv8.
    """
    st.title("SOLAR PANEL DAMAGE DETECTION")

    uploaded_file = st.file_uploader("Upload the image....", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)  # Convert to RGB

        boxes, labels, confidences = detect_objects(rgb_image)
        if len(boxes) > 0:
            for box, label, confidence in zip(boxes, labels, confidences):
                class_name = model.names[label]
                label_text = f"{class_name}: {confidence:.2f}"
                
                # Set the color based on confidence
                if confidence < 0.3:
                    color = (0, 0, 255)  # Red
                elif 0.3 <= confidence < 0.7:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 0)  # Green
                
                # Draw bounding box
                cv2.rectangle(rgb_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                
                # Calculate label position ensuring it's within the image boundaries
                text_x = int(box[0])
                text_y = int(box[1]) - 10
                if text_y < 10:  # Ensure the text is not out of the top boundary
                    text_y = int(box[1]) + 25

                draw_label(rgb_image, label_text, (text_x, text_y), color)

            st.image(rgb_image, channels="RGB", use_column_width=True)
        else:
            st.write("No objects detected.")

if __name__ == '__main__':
    main()
