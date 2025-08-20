import os
import numpy as np
import cv2

def preprocess_residual(res):
    res = cv2.resize(res, (224, 224), interpolation=cv2.INTER_LINEAR)
    res = res.astype(np.float32)
    res = res - 128.0
    res = res / 128.0
    res = np.transpose(res, (2, 0, 1))
    return res

# def preprocess_residual(res):
#     size = 20
#     res = (res * (127.5 / size)).astype(np.int32)
#     res += 128
#     res = (np.minimum(np.maximum(res, 0), 255)).astype(np.uint8)
#     res = cv2.resize(res, (224, 224), interpolation=cv2.INTER_LINEAR)
#     res = (res.astype(np.float32) / 255.0 - 0.5) / np.array([0.229, 0.224, 0.225])
#     res = np.transpose(res, (2, 0, 1)).astype(np.float32)
#     return res

def preprocess_motion_vector(mv):
    mv += 128
    mv = mv.astype(np.uint8)
    resized_mv = np.stack([cv2.resize(mv[..., i], (224, 224), interpolation=cv2.INTER_LINEAR) for i in range(2)], axis=0)
    resized_mv -= 128
    resized_mv = resized_mv.astype(np.float32)
    motion_strength = np.abs(resized_mv[0]) + np.abs(resized_mv[1])
    mask_mv = (motion_strength != 0).astype(np.uint8)  # 记录有发生移动的块
    return resized_mv,mask_mv

def read_residuals_and_motion_vectors(directory):
    res_list = []
    mv_list = []
    mask_mv_list = [] 

    try:
    # 获取目录中所有文件
        files = os.listdir(directory)
        files.sort()  # 按文件名排序，确保帧的顺序正确
    # 遍历文件
        for file_name in files:
            # print(file_name)
            process_file(directory, file_name, res_list, mv_list,mask_mv_list)
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    # 转换为NumPy数组
    res = np.array(res_list)
    mv = np.array(mv_list)
    mask_mv = np.array(mask_mv_list)
    mask_mv = mask_mv[:, np.newaxis, :, :]

    return res, mv, mask_mv

def process_file(directory, file_name, res_list, mv_list,mask_mv_list):
    file_path = os.path.join(directory, file_name)

    if file_name.startswith('res_') and file_name.endswith('.jpg'):
        # 读取残差数据
        residual = np.array(cv2.imread(file_path))
        residual = preprocess_residual(residual)
        res_list.append(residual)

    elif file_name.startswith('mv_') and file_name.endswith('.npy'):
        # 读取运动矢量数据
        motion_vector = np.load(file_path)
        # motion_vector = preprocess_motion_vector(motion_vector[:, :, :4])
        motion_vector,mask_mv = preprocess_motion_vector(motion_vector[:, :, :])
        mv_list.append(motion_vector)
        mask_mv_list.append(mask_mv)

def process_videos(video_list_file,videos_root_directory):

    data_dict_list = []
    # 读取 train.txt 文件中的视频目录列表
    with open(video_list_file, 'r') as file:
        video_dirs = [line.strip() for line in file.readlines()]

    # 遍历每个视频目录
    for video_dir in video_dirs:
        # 构建视频目录的完整路径
        full_video_dir = os.path.join(videos_root_directory, video_dir)

        # print(f"\nProcessing video in directory: {full_video_dir}")

        # 读取残差和运动矢量数据
        result_dict = read_residuals_and_motion_vectors(full_video_dir)
        data_dict_list.append(result_dict)

    return data_dict_list