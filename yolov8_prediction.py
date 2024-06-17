from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train/weights/best.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model([
    "Dataset/training_dataset/canada_receipts_3/images/train/2E3BCC32-8ACE-40C3-9438-C608A10251D0.jpeg",
    "Dataset/testing_dataset/Original Receipts/Fox Goes Free/IMG_2157 1.jpeg",
    "Dataset/testing_dataset/Synthetic Receipts/FoxGoesFree_HTML_Images/Fox_Trans: 63203.png"
])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
