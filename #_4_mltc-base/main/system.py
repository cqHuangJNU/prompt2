import os
import random
import shutil

import numpy as np
import torch
import transformers


class SYSTEM:
    """
    class SYSTEM: 解决定义和设置系统资源的任务.
    :param gpu_index: GPU的索引, 只有1块GPU索引是'0'.
    :param seed: 随机种子.
    """
    gpu_index = '0'
    seed = 1
    checkpoint_dir = "results"
    logging_dir = 'logs'

    @classmethod
    def clear_folder(cls, folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # 删除目录及其所有内容
        os.makedirs(folder_path)  # 创建目录

    @classmethod
    def config_system(cls) -> None:
        """
        设置GPU.
        设置随机种子.
        设置工作目录, 在Pycharm里有时候工作目录会是顶级目录, 需要额外设置
        :return: None
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = cls.gpu_index
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        transformers.set_seed(cls.seed)
        # if os.name is not "posix":
        #     current_dir = os.path.dirname(__file__)
        #     os.chdir(current_dir)
        cls.clear_folder(cls.checkpoint_dir)
        cls.clear_folder(cls.logging_dir)
