from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="canada_receipts.yaml",
    epochs=300,
    imgsz=1024)

