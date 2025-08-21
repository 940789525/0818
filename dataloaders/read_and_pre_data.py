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

def preprocess_motion_vector(mv, out_size=(224,224), mode='norm2', scale=128.0, eps=1e-6):
    """
    预处理 motion vector。

    参数:
      mv: np.ndarray, shape [H, W, C] (例如 [90,120,6])，dtype 任意数值型。
          已确认 mv[...,0] = dx, mv[...,1] = dy，且为有符号位移（像素/编码单位）。
      out_size: (W,H) 输出 spatial 大小，默认为 (224,224)。
      mode: 'norm2' | 'mag' | 'pol'
        'norm2' -> 返回 [2, H_out, W_out] (dx, dy) 归一化后；
        'mag'   -> 返回 [3, H_out, W_out] (dx, dy, magnitude)；
        'pol'   -> 返回 [4, H_out, W_out] (dx, dy, magnitude, angle)，
                   angle 已归一化到 [-1,1]，并在小 magnitude 处被置 0 以抑制噪声。
      scale: 用于归一化的常数（将 dx,dy / scale），默认 128。根据你数据可调整。
      eps: 用于判定“近零运动”的阈值。

    返回:
      resized_mv: np.float32, shape [C_out, H_out, W_out]
      mask_mv:    np.uint8,   shape [H_out, W_out]  (motion presence)
    """

    # --- 1. 安全提取 dx, dy ---
    # 确认 mv 至少有两个通道
    assert mv.ndim == 3 and mv.shape[2] >= 2, "mv must have shape [H, W, >=2]"
    dx = mv[..., 0].astype(np.float32)
    dy = mv[..., 1].astype(np.float32)

    # --- 2. 归一化（固定尺度，比直接按最大值更稳定） ---
    dxn = dx / float(scale)
    dyn = dy / float(scale)

    # --- 3. 计算极坐标分量（magnitude, angle） ---
    mag = np.sqrt(dxn * dxn + dyn * dyn).astype(np.float32)   # >=0
    # angle in [-pi, pi], normalized to [-1, 1]
    ang = np.arctan2(dyn, dxn).astype(np.float32) / np.pi

    # 把 angle 在极小 magnitude 处置 0，避免噪声
    ang = np.where(mag > eps, ang, 0.0).astype(np.float32)

    # --- 4. resize 每个通道到 out_size (W,H) ---
    W_out, H_out = out_size
    # 注意：cv2.resize 输入为 (width, height)
    to_resize = []
    if mode == 'norm2':
        chs = [dxn, dyn]
    elif mode == 'mag':
        chs = [dxn, dyn, mag]
    elif mode == 'pol':
        chs = [dxn, dyn, mag, ang]
    else:
        raise ValueError("mode must be one of 'norm2','mag','pol'")

    for ch in chs:
        # cv2.resize expects shape (H, W)
        resized = cv2.resize(ch, (W_out, H_out), interpolation=cv2.INTER_LINEAR)
        to_resize.append(resized.astype(np.float32))

    resized_mv = np.stack(to_resize, axis=0)   # [C_out, H_out, W_out]

    # --- 5. mask: 基于 magnitude 或 dx/dy 判断是否有运动 ---
    # 使用 magnitude（具有旋转不变性）；若没有 mag 通道则用 abs(dx)+abs(dy)
    if mode in ('mag', 'pol'):
        motion_strength = resized_mv[2]   # mag channel
    else:
        motion_strength = np.abs(resized_mv[0]) + np.abs(resized_mv[1])

    # 二值 mask（uint8），阈值可以微调
    mask_mv = (motion_strength > (eps * 10)).astype(np.uint8)

    return resized_mv, mask_mv

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
        # motion_vector.shape=[90,120,6]  其中 motion_vector[:,:,0:2] 表示参考前一帧的运动矢量  motion_vector[]
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