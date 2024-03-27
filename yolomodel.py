from ultralytics import YOLO
import cv2
import torch
import os
import glob
import argparse

# Load a model
model = YOLO("yolov8n-pose.pt")  # load an official model


# Predict with the model
def predict(image_path):
    results = model(
        image_path, show_labels=False, show_conf=False, show_boxes=False
    )  # predict on an image
    keypoints = (
        results[0].keypoints.xyn.cpu().numpy()
    )  # get keypoints (x, y) coordinates
    humant_pose = keypoints[0, :, :]  # get the first person's keypoints
    humant_pose = humant_pose.reshape(-1)  # reshape to (N,) format
    str = ""
    for i in range(len(humant_pose)):
        str += f"{humant_pose[i]}"
        if i != len(humant_pose) - 1:
            str += ","
    if len(humant_pose) < 17 * 2:
        print("Can't collect 17 keypoint")
        return None
    return str + "\n"


def write_data(dataX, dataY, file_path):
    if dataY != "":
        if len(dataX) == 32:
            with open(f"{file_path}X.txt", "a") as file:
                file.writelines(dataX)
                # file.writelines(f"dem 3x2 : {len(dataX)} \n")
            with open(f"{file_path}Y.txt", "a") as file:
                file.writelines(f"{dataY}\n")
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


def get_video_frame(video_path, label, file_path, n_steps=32):
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
    time_in_seconds = duration_seconds / (n_steps + 1)

    frame_ = n_steps
    out_X = []
    while frame_:

        time_in_milliseconds = (n_steps - frame_) * time_in_seconds * 1000
        # print(f" Collect frame {n_steps-frame_}")

        # Di chuyển tới thời điểm đã chỉ định
        cap.set(cv2.CAP_PROP_POS_MSEC, time_in_milliseconds)

        success, frame = cap.read()

        if not success:
            print("Can't read frame!")
            break

        with torch.no_grad():
            output = predict(frame)
            if output == None:
                print("Can't collect 17 keypoint")
                return False
            out_X.append(output)

        frame_ -= 1

    # Đóng video
    cap.release()
    return write_data(out_X, label, file_path)


# Đường dẫn đến video
# get_video_frame("tricep pushdown_49.mp4",1,"data")


def reprocess(folder_path, LABELS, current_time):
    count = 0
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    for i in range(0, len(LABELS)):
        file_path = parent_directory + f"/{folder_path}/{LABELS[i]}/"
        print(file_path)
        video_files = glob.glob(os.path.join(file_path, "*.mp4"))

        for video_file in video_files:
            print(video_file, i)
            if get_video_frame(video_file, i + 1, f"{current_time}data") == True:
                count += 1
    return count


def train(dataset_folder, LABELS, current_time):
    print("Training")
    reprocess(dataset_folder, LABELS, current_time)
    file_pathx = f"{current_time}dataX.txt"
    file_pathy = f"{current_time}dataY.txt"
    file_path_trainx = f"{current_time}dataX_train.txt"
    file_path_trainy = f"{current_time}dataY_train.txt"

    file_path_testx = f"{current_time}dataX_test.txt"
    file_path_testy = f"{current_time}dataY_test.txt"

    n_steps = 32
    split_ratio = 0.9
    read_filesx = open(file_pathx, "r").readlines()
    print(len(read_filesx))
    read_filesy = open(file_pathy, "r").readlines()
    print(len(read_filesy))
    Y = {}
    for read_file in read_filesy:
        read_file = read_file.strip()
        if read_file not in Y:
            Y[read_file] = 1
        else:
            Y[read_file] += 1
    t = 0
    for key in Y:
        for i in range(0, Y[key]):
            if i <= int(Y[key] * (1 - split_ratio)):
                with open(file_path_testy, "a") as file:
                    file.write(key + "\n")
                for j in range(0, n_steps):
                    with open(file_path_testx, "a") as file:
                        file.write(read_filesx[t])
                    t += 1
            else:
                with open(file_path_trainy, "a") as file:
                    file.write(key + "\n")
                for j in range(0, n_steps):
                    with open(file_path_trainx, "a") as file:
                        file.write(read_filesx[t])
                    t += 1
    return file_path_trainx, file_path_trainy, file_path_testx, file_path_testy


def test(dataset_folder, LABELS, current_time):
    print("Testing")
    reprocess(dataset_folder, LABELS, current_time)
    file_path_trainx = f"{current_time}data_testX.txt"
    file_path_trainy = f"{current_time}data_testY.txt"

    file_path_testx = f"{current_time}data_testX.txt"
    file_path_testy = f"{current_time}data_testY.txt"
    return file_path_trainx, file_path_trainy, file_path_testx, file_path_testy
