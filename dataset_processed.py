# 定义下载相关函数
import os
import subprocess
import sys

dj_path = '/home/Mount_8T/Modelscope_sora_solution5/data_juicer' # assign absolute path

# config_path = os.path.join(os.getcwd(), 'demo_config.yaml')
config_path = '/home/Mount_8T/Modelscope_sora_solution5/dataset_processed.yaml' # assign absolute path

# !cd {dj_path} && PATHPATH=./ python tools/process_data.py --config {config_path}
# 构建命令
command = ['python', 'tools/process_data.py', '--config', config_path]
# 切换工作目录并执行命令
print('start')
os.chdir(dj_path)
# result = subprocess.run(command, capture_output=True, text=True) # not output log
result = subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr, text=True)  # output log

# 打印输出和错误信息
print("Output:", result.stdout)
print("Error:", result.stderr)
