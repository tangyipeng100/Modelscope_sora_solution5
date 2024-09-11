from loguru import logger
import sys

# 替换下面的路径为您的 data_juicer 模块的绝对路径
path_to_data_juicer = "/home/Mount_2_6T/dj_sora_challenge/toolkit/data-juicer"
sys.path.append(path_to_data_juicer)

from data_juicer.config import init_configs
from data_juicer.core import Executor


@logger.catch(reraise=True)
def main():
    cfg = init_configs()
    if cfg.executor_type == 'default':
        executor = Executor(cfg)
    elif cfg.executor_type == 'ray':
        from data_juicer.core.ray_executor import RayExecutor
        executor = RayExecutor(cfg)
    executor.run()


if __name__ == '__main__':
    main()
