from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model

# Predict with the model
results = model('4470ea421234a1d0aa58cab42ecd1469.jpg')  # predict on an image
keypoints = results[0].keypoints.xyn.cpu().numpy()  # get keypoints (x, y) coordinates
humant_pose = keypoints[0,:,:]  # get the first person's keypoints
humant_pose = humant_pose.reshape(-1)  # reshape to (N,) format
print(humant_pose)
a =[[0.4640, 0.2605],
         [0.4893, 0.2641],
         [0.4685, 0.2507],
         [0.5131, 0.3026],
         [0.0000, 0.0000],
         [0.5115, 0.4129],
         [0.4055, 0.3177],
         [0.6076, 0.5038],
         [0.3000, 0.2058],
         [0.7331, 0.5796],
         [0.3025, 0.1201],
         [0.4255, 0.5844],
         [0.3991, 0.5562],
         [0.3781, 0.7784],
         [0.5616, 0.7531],
         [0.3470, 0.9304],
         [0.6936, 0.9053]]
