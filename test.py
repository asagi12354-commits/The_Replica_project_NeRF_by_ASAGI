import torch
from colorama.win32 import winapi_test
from taichi.examples.real_func.rendering.cornell_box import image_pixels
from torch.utils.tensorboard.summary import make_video

torch._dynamo.disable()  # 关掉冲突的编译器
# 系统路径操作库（用来拼接文件路径）
import os
# 读取json文件（数据集的相机参数都在这里）
import json
# 数值计算库（处理图片、矩阵必备）
import numpy as np
# 图像处理库（缩放图片用）
import cv2
# 读取图片文件
import imageio
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim


from utils import  DatabaseProvider ,NeRFDataset #导入初始化类
from utils import  Embedder,ViewDependentHead,NoViewDirHead
from utils import NeRF
from utils import sample_rays,sample_viewdirs,predict_to_rgb,sample_pdf
from utils import render_rays








