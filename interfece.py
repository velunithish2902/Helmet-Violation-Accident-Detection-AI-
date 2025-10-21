# interfece.py
from ultralytics import YOLO
import cv2
import os
from glob import glob

# Path to your trained YOLO model
MODEL_PATH = r"C:\Users\Administrator\nith\best (1).pt"

# Folder with images (can be single image too)
IMAGE_FOLDER = r"C:\Users\Administrator\nith\photo\no helment3.jpg"
OUTPUT_FOLDER = r"C:\Users\Administrator\nith\output"

CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to show box

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

def detect_image(input_path, out_path=None, conf=CONFIDENCE_THRESHOLD):
    """
    Run YOLO detection on a single image.
    Draws bounding boxes and saves output if out_path is provided.
    """
    results = model.predict(source=input_path, conf=conf, save=False, imgsz=640)
    r = results[0]
    img = r.orig_img.copy()

    for box, cls, cf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {cf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if out_path:
        cv2.imwrite(out_path, img)

    return r

if __name__ == "__main__":
    # Detect all images in folder (jpg, png)
    image_files = glob(os.path.join(IMAGE_FOLDER, "*.*"))
    supported_ext = (".jpg", ".jpeg", ".png")
    image_files = [f for f in image_files if f.lower().endswith(supported_ext)]

    if not image_files:
        print("No images found in folder:", IMAGE_FOLDER)
    else:
        for img_path in image_files:
            out_name = os.path.join(OUTPUT_FOLDER, os.path.basename(img_path))
            res = detect_image(img_path, out_name)
            print(f"Processed: {img_path}")
            print(f" - Detections: {len(res.boxes)}")
            if len(res.boxes) > 0:
                print(f" - Average confidence: {res.boxes.conf.mean().item():.2f}")
            print(f" - Saved output: {out_name}\n")

    print("All images processed. Check the output folder:", OUTPUT_FOLDER)
