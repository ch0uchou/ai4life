from ultralytics import YOLO
import cv2
import torch
import os
import glob
import argparse


parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--data', default='./NEWAI4LIFE2024-DATA', type=str, help='Dataset path', required=False)   
args = parser.parse_args()
dataset_folder = args.data

LABELS = [
  "russian twist",
  "tricep dips",
  "t bar row",
  "squat",
  "shoulder press",
  "romanian deadlift",
  "push-up",
  "plank",
  "leg extension",
  "leg raises",
  "lat pulldown",
  "incline bench press",
  "tricep pushdown",
  "pull up",
  "lateral raise",
  "hammer curl",
  "decline bench press",
  "hip thrust",
  "bench press",
  "chest fly machine",
  "deadlift",
  "barbell biceps curl"
]

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
  return str + "\n"

def write_data(dataX, dataY, file_path):
  if dataY != "":
    if len(dataX) == 32:
      with open(f'{file_path}X.txt', 'a') as file:
        file.writelines(dataX)
        # file.writelines(f"dem 3x2 : {len(dataX)} \n")
      with open(f'{file_path}Y.txt', 'a') as file:
        file.writelines(f'{dataY}\n')
      return True
    else:
      print("Can't collect 17 keypoint")
      return False
  else:
    if len(dataX) == 32:
      return dataX
    else:
      print("Can't collect 17 keypoint")
      return False

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
  out_X = []
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
      out_X.append(output)

    frame_ -= 1

  # Đóng video
  cap.release()
  return write_data(out_X, label, file_path)

# Đường dẫn đến video
# get_video_frame("tricep pushdown_49.mp4",1,"data")

def reprocess(folder_path):
  count = 0
  for i in range (0, len(LABELS)):
    file_path = f"{folder_path}/{LABELS[i]}/"
    print(file_path)
    video_files = glob.glob(os.path.join(file_path, '*.mp4'))

    for video_file in video_files:
      print(video_file, i)
      if get_video_frame(video_file, i + 1, "data") == True:
        count +=1
  return count

reprocess(dataset_folder)