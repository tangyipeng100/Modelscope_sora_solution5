# 定义下载相关函数
import os
import subprocess


# def aria2(url, filename, d):
#     !aria2c --console-log-level=error -c -x 16 -s 16 {url} -o {filename} -d {d}

def aria2(url, filename, directory):
    command = [
        "aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16",
        url, "-o", filename, "-d", directory
    ]
    # 执行命令，并捕获输出
    result = subprocess.run(command, capture_output=True, text=True)

    # 打印输出结果的标准输出和标准错误
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)


def download_from_oss(url, filename, save_dir):
    url_prefix = {
        "cn-shanghai": "http://pai-vision-data-sh.oss-cn-shanghai-internal.aliyuncs.com",
        "cn-hangzhou": "http://pai-vision-data-hz2.oss-cn-hangzhou-internal.aliyuncs.com",
        "cn-shenzhen": "http://pai-vision-data-sz.oss-cn-shenzhen-internal.aliyuncs.com",
        "cn-beijing": "http://pai-vision-data-bj.oss-cn-beijing-internal.aliyuncs.com",
        "cn-wulanchabu": "https://pai-vision-data-wlcb.oss-cn-wulanchabu-internal.aliyuncs.com",
        "ap-southeast-1": "http://pai-vision-data-ap-southeast.oss-ap-southeast-1-internal.aliyuncs.com"
    }
    dsw_region = os.environ.get("dsw_region")

    prefix = url_prefix[
        dsw_region] if dsw_region in url_prefix else "http://pai-vision-data-sh.oss-cn-shanghai.aliyuncs.com"
    url = os.path.join(prefix, url, filename)
    print(url)
    print(dsw_region)

    print(f'Download from {url}')
    # !aria2c --console-log-level=error -c -x 16 -s 16 {url} -o {filename} -d {save_dir}
    aria2(url, filename, save_dir)
    if filename.endswith('.tar.gz') or filename.endswith('.tar'):
        saved_filepath = os.path.join(save_dir, filename)
        print('Start Unzipping...')
        # !tar -xf $saved_filepath -C $save_dir
        command = ['tar', '-xf', saved_filepath, '-C', save_dir]
        subprocess.run(command)
        print('Done')


def check_files_exists_and_download(data_path):
    download_filenames = [
        "PixArt-XL-2-512x512.tar",
        "easyanimate_mm_16x256x256_pretrain.safetensors",
        "easyanimate_mm_16x512x512_pretrain.safetensors",
        "vbench_models.tar.gz"
    ]
#train_infer_evaluate/pretrained_models/Diffusion_Transformer
    filenames = [
        os.path.join(data_path, f"train_infer_evaluate/pretrained_models/Diffusion_Transformer/PixArt-XL-2-512x512.tar"),
        os.path.join(data_path, f"train_infer_evaluate/pretrained_models/Motion_Module/easyanimate_v1_mm.safetensors"),
        os.path.join(data_path,
                     f"train_infer_evaluate/pretrained_models/Motion_Module/easyanimate_mm_16x512x512_pretrain.safetensors"),
        os.path.join(data_path, f"train_infer_evaluate/evaluation/vbench_models.tar.gz")
    ]

    for download_filename, filename in zip(download_filenames, filenames):
        if os.path.exists(filename):
            print('Exists. ', filename)
            continue
        save_dir = os.path.dirname(filename)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Start Downloading: {download_filename} to {save_dir}")
        download_from_oss('aigc-data/easyanimate/models', download_filename, save_dir)


def download_dj_preprocess_model(op_name, dj_path):
    dj_model_dict = {
        'video_ocr_area_ratio_filter': ['craft_mlt_25k.pth'],
        'video_aesthetics_filter': ['models_aesthetic.tar.gz'],
        'video_frames_text_similarity_filter': ['models_clip.tar.gz'],
        'video_nsfw_filter': ['models_nsfw.tar.gz'],
        'video_watermark_filter': ['models_watermark.tar.gz'],
        'video_tagging_from_frames_mapper': ['ram_plus_swin_large_14m.pth', 'models_bert.tar.gz'],
        'video_captioning_from_frames_mapper': ['models_blip.tar.gz'],
        'video_captioning_from_video_mapper': ['models_video_blip.tar.gz']
    }

    if op_name == 'video_ocr_area_ratio_filter':
        # filenames = ['/root/.EasyOCR/model/craft_mlt_25k.pth']
        filenames = ['./craft_mlt_25k.pth']
    else:
        filenames = [os.path.join(dj_path, filename) for filename in dj_model_dict[op_name]]

    for download_filename, filename in zip(dj_model_dict[op_name], filenames):
        if os.path.exists(filename):
            print('Exists. ', filename)
            continue
        save_dir = os.path.dirname(filename)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Start Downloading: {download_filename} to {save_dir}")
        download_from_oss('aigc-data/easyanimate/models/preprocess', download_filename, save_dir)

# 下载数据到指定目录 5w原始数据，下载&解压 约 1h
data_path = os.path.join(os.getcwd(),'dj_sora_challenge/input2')
download_from_oss('aigc-data/easyanimate/data', 'input-data.tar.gz', data_path)

# 数据预处理算子下载
dj_path = os.path.join(os.getcwd(),'dj_sora_challenge/toolkit/data-juicer')
download_dj_preprocess_model('video_ocr_area_ratio_filter', dj_path)
download_dj_preprocess_model('video_aesthetics_filter', dj_path)
download_dj_preprocess_model('video_nsfw_filter', dj_path)
download_dj_preprocess_model('video_frames_text_similarity_filter', dj_path)
download_dj_preprocess_model('video_watermark_filter', dj_path)
download_dj_preprocess_model('video_tagging_from_frames_mapper', dj_path)
download_dj_preprocess_model('video_captioning_from_frames_mapper', dj_path)
download_dj_preprocess_model('video_captioning_from_video_mapper', dj_path)