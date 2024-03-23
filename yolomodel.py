from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model

# Predict with the model
results = model(source="tricep pushdown_49.mp4")  # predict on an image
print(results[0].keypoints.xyn.cpu().numpy())
# keypoints = results[0].keypoints.xyn.cpu().numpy()  # get keypoints (x, y) coordinates
# if keypoints is None:
#   print('No person detected in the image')
#   return None
# humant_pose = keypoints[0,:,:]  # get the first person's keypoints
# humant_pose = humant_pose.reshape(-1)  # reshape to (N,) format
# return humant_pose

# def write_data(inputX, inputY):

# a = [1, 2, 3]
# image_path = '4470ea421234a1d0aa58cab42ecd1469.jpg'
# with open(f'Xtest.txt', 'a') as file:
#   str = ''
#   for i in range(len(a)):
#     str += f'{a[i]}'
#     if i != len(a) - 1:
#       str += ','
#   file.write(str + "\n")


