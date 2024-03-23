from ultralytics import YOLO
import cv2
import torch

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model

# Predict with the model
def predict(image_path):
  results = model(image_path)  # predict on an image
  keypoints = results[0].keypoints.xyn.cpu().numpy()  # get keypoints (x, y) coordinates
  if keypoints is None:
    print('No person detected in the image')
    return None
  humant_pose = keypoints[0,:,:]  # get the first person's keypoints
  humant_pose = humant_pose.reshape(-1)  # reshape to (N,) format
  str = ''
  for i in range(len(humant_pose)):
    str += f'{humant_pose[i]}'
    if i != len(humant_pose) - 1:
      str += ','
  return str

# def write_data(inputX,):

def get_video_frame(video_path, label, file_path, n_steps = 32):
  # Mở video
  cap = cv2.VideoCapture(video_path)

  # Kiểm tra video có mở thành công không
  if not cap.isOpened():
      print("Can't open video!")
      return None

  # Lấy tổng số frame và số frame trên mỗi giây
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  # Tính toán thời lượng của video
  duration_seconds = frame_count / fps

  # Lấy thời lượng của video
  time_in_seconds = (duration_seconds / (n_steps + 1))

  frame_ = n_steps
  while frame_:

    time_in_milliseconds = (n_steps-frame_)*time_in_seconds*1000
    # print(f" Collect frame {n_steps-frame_}")

    # Di chuyển tới thời điểm đã chỉ định
    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_milliseconds)

    success, frame = cap.read()

    if not success:
      print("Can't read frame!")
      break

    with torch.no_grad():
      output= predict(frame)
      print(output)

    frame_ -= 1

  # Đóng video
  cap.release()

get_video_frame("tricep pushdown_49.mp4",1,"data")
