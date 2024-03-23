from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model('4470ea421234a1d0aa58cab42ecd1469.jpg')  # predict on an image
print(f'{results.__dir__}')  # print results