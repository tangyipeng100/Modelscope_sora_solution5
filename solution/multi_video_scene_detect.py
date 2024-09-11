import scenedetect
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import os

# 定义视频文件目录和文件名模式，运行脚本前需要先修改以下目录
output_dir = 'E:/input/' # 保存视频的目录
video_dir = '../../input' # 视频文件目录
video_prefix = 'dj_video_' # 视频文件名前缀
video_suffix = '.mp4' # 视频文件名后缀
video_count = 49999 # 视频文件数量

def process_video(video_path, video_name):
    # 创建视频管理器
    video_manager = VideoManager([video_path])

    # 创建场景管理器
    scene_manager = SceneManager()

    # 添加内容检测器到场景管理器
    scene_manager.add_detector(ContentDetector(threshold=30.0))

    # 启动视频管理器
    video_manager.start()

    # 检测场景
    scene_manager.detect_scenes(frame_source=video_manager)

    # 获取检测到的场景列表
    scene_list = scene_manager.get_scene_list()

    # 打印场景信息
    print(f'{len(scene_list)} scenes detected in {video_name}!')

    # 保存大于2秒的场景
    saved_scenes = 0
    for i, scene in enumerate(scene_list):
        start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
        if (end_frame - start_frame) / video_manager.get(cv2.CAP_PROP_FPS) > 2:
            save_scene(video_path, scene, video_name, saved_scenes)
            saved_scenes += 1

    # 如果没有检测到大于2秒的场景，则保存原始视频
    if saved_scenes == 0:
        save_scene(video_path, [(0, int(video_manager.get(cv2.CAP_PROP_FRAME_COUNT)))], video_name, 0)

    # 释放视频管理器资源
    video_manager.release()

def save_scene(video_path, scene, video_name, index):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取场景起始和结束帧
    try:
        start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
    except:
        start_frame, end_frame = scene[0][0], scene[0][1]

    # 定义视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_filename = f'scene_detect/{video_name}_{index}.mp4'
    out_filepath = os.path.join(output_dir, out_filename) #
    out = cv2.VideoWriter(out_filepath, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # 设置视频捕捉器的位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 读取并写入帧
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()

# 遍历所有视频文件
for i in range(1, video_count + 1):
    video_name = f'{video_prefix}{i:05d}'
    video_path = os.path.join(video_dir, f'{video_name}{video_suffix}')
    process_video(video_path, video_name)
