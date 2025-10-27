from ultralytics import YOLO


# Load YOLOv8 model (you can use yolov8n.pt or any variant)
model = YOLO("yolov8n.pt")  # change if you have yolov8s/yolov8m

model.train(
    data="/content/merged_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="helmet_project",
    project="/content/runs/detect",
    augment=True,      # enables flips, rotations, brightness etc.
    lr0=0.001,
    patience=30,       # stop early if no improvement



)