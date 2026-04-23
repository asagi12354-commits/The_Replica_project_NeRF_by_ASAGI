import torch
torch._dynamo.disable()  # 关掉PyTorch动态编译器，避免和NeRF训练冲突报错
# 系统路径操作库：用于拼接数据集的文件路径，跨系统不报错
import os
# JSON读取库：数据集的相机位姿、视角、图片路径全部存在JSON里
import json
# 数值计算核心库：处理图片矩阵、相机参数、数值运算必备
import numpy as np
# 开源图像处理库：用于缩放图片分辨率，加速训练
import cv2
# 图片读取库：高效读取PNG/JPG图片，支持透明通道
import imageio

# PyTorch神经网络核心库：搭建NeRF网络层
import torch.nn as nn
# PyTorch优化器库：训练网络时更新参数
import torch.optim as optim

from tqdm import tqdm



# ===================================================================================
# 数据集加载器：专门负责读取NeRF官方lego数据集
# 核心功能：读JSON相机参数 → 读图片 → 算焦距 → 处理透明背景 → 统一数据格式
# ===================================================================================
class DatabaseProvider:
    # 构造函数：创建对象时自动执行，完成所有数据初始化
    # root: 数据集根文件夹路径
    # transforms_file: 存储相机参数的JSON文件名
    # half_resolution: 是否将图片分辨率缩小一半（加速训练）
    def __init__(self, root, transforms_file, half_resolution=False):
        # --------------------- 第一步：读取并解析相机参数JSON文件 ---------------------
        # 拼接JSON文件完整路径，只读模式打开
        with open(os.path.join(root, transforms_file), "r") as f:
            # 将JSON文本加载为Python字典，方便取值
            self.meta = json.load(f)
            # 保存数据集根路径到对象属性，后续读图片用
            self.root = root
            # 提取所有帧的信息：每帧包含图片路径+相机4x4位姿矩阵
            self.frames = self.meta["frames"]
            # 初始化空列表：存储所有图片的像素数据
            self.images = []
            # 初始化空列表：存储所有图片对应的相机位姿矩阵
            self.poses = []
            # 从JSON中提取相机水平视场角，用于计算相机焦距（NeRF核心参数）
            self.camera_angles_x = self.meta["camera_angle_x"]

        # --------------------- 第二步：循环遍历所有帧，读取图片+相机位姿 ---------------------
        # 遍历每一帧的信息
        for frame in self.frames:
            # 拼接单张图片的完整路径，JSON里的路径没有后缀，手动加.png
            image_file = os.path.join(root, frame["file_path"] + '.png')
            # 读取图片 → 转换为numpy数组（形状：H, W, 4，RGBA四通道）
            image = imageio.imread(image_file)

            # 如果开启半分辨率，将图片宽高各缩小50%
            if half_resolution:
                image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            # 将处理好的图片加入列表
            self.images.append(image)
            # 将当前帧的相机4x4位姿矩阵加入列表
            self.poses.append(frame["transform_matrix"])

        # --------------------- 第三步：列表转矩阵，数据归一化 ---------------------
        # 将所有相机位姿堆叠为三维矩阵：[图片数量, 4, 4]
        self.poses = np.stack(self.poses)
        # 将所有图片堆叠为四维矩阵：[图片数量, H, W, 4] → 除以255归一化到0~1 → 转float32
        self.images = (np.stack(self.images) / 255.).astype(np.float32)

        # --------------------- 第四步：提取图片尺寸 ---------------------
        # 图片矩阵形状：[N, 高度, 宽度, 通道数]
        self.width = self.images.shape[2]    # 图片宽度 W
        self.height = self.images.shape[1]   # 图片高度 H

        # --------------------- 第五步：计算相机焦距（NeRF渲染必备核心参数） ---------------------
        # 几何公式：由水平视场角计算焦距，标准相机针孔模型
        self.focal = 0.5 * self.width / np.tan(0.5 * self.camera_angles_x)

        # --------------------- 第六步：处理透明背景（RGBA→RGB，透明部分变白色） ---------------------
        # 提取透明通道Alpha：第4个通道，控制像素透明度
        alpha = self.images[..., [3]]
        # 提取RGB颜色通道：前3个通道
        rgb = self.images[..., :3]
        # 透明混合公式：透明区域(alpha=0)显示白色，不透明区域显示原图
        self.images = rgb * alpha + (1 - alpha)

# ===================================================================================
# 输出头A：无视角依赖
# 适用：哑光物体（颜色不随观察角度变化）
# 输入：3D点位置特征
# 输出：RGB颜色 + 体密度σ
# ===================================================================================
class NoViewDirHead(nn.Module):
    def __init__(self,ninput,noutput):
        super().__init__()
        # 单层全连接层：输入位置特征 → 输出4个值（RGB3通道 + 密度1通道）
        self.head = nn.Linear(ninput,noutput)

    def forward(self, x, view_dirs):
        # 前向传播：输入特征 → 线性变换输出
        x = self.head(x)
        # 拆分输出：前3个通道 = RGB颜色值
        rgb = x[..., :3]
        # 拆分输出：第4个通道 = 体密度σ → ReLU保证密度≥0（物理意义：密度不能为负）
        sigma = x[..., 3:].relu()
        # 返回：预测的颜色、密度
        return rgb, sigma

# ===================================================================================
# 输出头B：视角依赖（NeRF真实感核心！）
# 适用：金属/玻璃/皮肤（反光、颜色随观察角度变化）
# 输入：3D点位置特征 + 视角方向特征
# 输出：体密度σ + 视角相关RGB颜色
# ===================================================================================
class ViewDependentHead(nn.Module):
    def __init__(self,ninput,nview):
        super().__init__()
        # 全连接层：对位置特征做非线性变换
        self.feature = nn.Linear(ninput, ninput)
        # 全连接层：直接输出体密度（1个值）
        self.alpha = nn.Linear(ninput, 1)
        # 全连接层：拼接【位置特征+视角特征】→ 降维提取视角相关特征
        self.view = nn.Linear(ninput + nview, ninput // 2)
        # 全连接层：最终输出RGB颜色（3个值）
        self.rgb = nn.Linear(ninput // 2, 3)

    def forward(self,x,view_dirs):
        # x：主干网络输出的3D点位置特征
        # view_dirs：编码后的视角方向特征
        # 1. 变换位置特征
        feature = self.feature(x)
        # 2. 预测体密度 → ReLU保证≥0
        sigma = self.alpha(x).relu()
        # 3. 核心操作：拼接位置特征 + 视角方向（颜色随角度变化的关键）
        feature = torch.cat([feature, view_dirs], dim=-1)
        # 4. 视角特征变换 + ReLU激活
        feature = self.view(feature).relu()
        # 5. 预测RGB颜色 → Sigmoid归一化到0~1（符合图片像素范围）
        rgb = self.rgb(feature).sigmoid()
        # 返回：视角相关颜色、体密度
        return rgb, sigma

# ===================================================================================
# 位置/视角编码器：NeRF核心组件
# 作用：将低维3D坐标 → 高维正弦余弦特征，让网络学习高频细节（物体边缘/纹理）
# ===================================================================================
class Embedder(nn.Module):
    def __init__(self,encoding_dim):
        super().__init__()
        # 编码层数：控制高频细节的丰富程度
        self.encoding_dim=encoding_dim

    def forward(self,x):
        # 初始化结果列表：先放入原始坐标
        res=[x]
        # 循环进行正弦/余弦编码，频率指数递增
        for i in range(self.encoding_dim):
            for fn in [torch.sin,torch.cos]:
                # 编码公式：sin(2^i * x) / cos(2^i * x)
                res.append(fn(2.**i * x))
        # 拼接所有编码特征，输出高维向量
        return  torch.cat(res,dim=-1)

# ===================================================================================
# NeRF 主网络
# 结构：8层MLP + 跳跃连接 + 位置编码 + 视角编码
# 功能：输入3D空间点 + 观察视角 → 输出RGB颜色 + 体密度σ
# ===================================================================================
class NeRF(nn.Module):
    # 初始化函数：搭建网络所有层
    # x_pedim：位置编码层数（控制物体细节）
    # nwidth：每层网络神经元数量（网络宽度）
    # ndeepth：网络总层数
    # view_pedim：视角编码层数（控制反光真实感）
    def __init__(self, x_pedim=10, nwidth=256, ndeepth=8, view_pedim=4):
        super().__init__()  # 固定写法：初始化父类

        # ===================== 1. 计算位置编码后的输入维度 =====================
        # 3维坐标编码后维度公式：(2*L + 1)*3  L=编码层数
        xdim = (x_pedim * 2 + 1) * 3

        # 存储所有MLP层
        layers = []
        # 定义每一层的输入维度：默认全部为nwidth(256)
        layers_in = [nwidth] * ndeepth
        # 第一层输入：编码后的高维位置特征
        layers_in[0] = xdim
        # 第5层输入：256 + 原始编码特征（跳跃连接，防止梯度消失）
        layers_in[5] = nwidth + xdim

        # 循环创建8层全连接层（MLP主干网络）
        for i in range(ndeepth):
            # 添加线性层：输入维度 → 256
            layers.append(nn.Linear(layers_in[i], nwidth))

        # ===================== 2. 判断是否使用视角依赖 =====================
        if view_pedim > 0:
            # 计算视角编码后的维度
            view_dim = (view_pedim * 2 + 1) * 3
            # 视角编码器（修复原代码拼写BUG：view_emded → view_embed）
            self.view_embed = Embedder(view_pedim)
            # 使用带视角的输出头（真实感更强）
            self.head = ViewDependentHead(nwidth, view_dim)
        else:
            # 不使用视角，使用简易输出头
            self.head = NoViewDirHead(nwidth, 4)

        # 位置编码器：给3D坐标添加细节
        self.xembed = Embedder(x_pedim)
        # 将所有层打包为序列网络
        self.layers = nn.Sequential(*layers)

    # ===================== 前向传播：数据在网络中的流动逻辑 =====================
    # x：3D空间点坐标 (x,y,z) 形状：[光线数, 采样点数, 3]
    # view_dirs：观察视角方向 形状：[光线数, 采样点数, 3]
    def forward(self, x, view_dirs):
        # 记录输入坐标形状，用于后续视角特征广播对齐
        xshape = x.shape

        # 1. 3D位置编码：低维坐标 → 高维特征
        x = self.xembed(x)

        # 2. 如果启用视角编码，对视角方向做编码
        if self.view_embed is not None:
            # 视角编码：3维方向 → 高维特征
            view_dirs = self.view_embed(view_dirs)

        # 3. 备份原始编码特征（跳跃连接专用）
        raw_x = x

        # 4. 遍历8层MLP网络
        for i, layer in enumerate(self.layers):
            # 线性变换 + ReLU激活
            x = torch.relu(layer(x))
            # 核心：第5层（索引i=4）执行跳跃连接，拼接原始位置特征
            if i == 4:
                x = torch.cat([x, raw_x], axis=-1)

        # 5. 经过输出头，得到最终颜色+密度
        return self.head(x, view_dirs)

# ===================================================================================
# NeRF 数据集类：核心！将图片像素 → 相机光线（起点+方向）
# 功能：生成训练用光线、中心裁剪加速收敛、批量采样光线
# ===================================================================================
class NeRFDataset:
    def __init__(self,provider:DatabaseProvider,batch_size,device="cuda"):
        # 把DatabaseProvider处理好的数据全部迁移过来
        self.images = provider.images  # 所有训练图片 [N, H, W, 3]
        self.poses = provider.poses    # 所有相机位姿 [N, 4, 4]
        self.focal = provider.focal    # 相机焦距
        self.width = provider.width    # 图片宽度
        self.height = provider.height  # 图片高度
        self.batch_size = batch_size   # 每批次训练的光线数量
        self.num_image = len(self.images)  # 总图片数
        # 训练超参：前500轮只训练图片中心区域，加速收敛
        self.precrop_iters = 500
        # 中心区域占比：50%
        self.precrop_frac = 0.5
        self.niter = 0  # 记录当前训练迭代次数
        self.device = device  # 运行设备：GPU/CPU
        self.initialize()  # 初始化：生成所有图片的光线

    def initialize(self):
        # 生成宽度方向像素坐标：0,1,2...W-1
        warange = torch.arange(self.width, dtype=torch.float32, device=self.device)
        # 生成高度方向像素坐标：0,1,2...H-1
        harange = torch.arange(self.height, dtype=torch.float32, device=self.device)
        # 生成像素网格坐标：x=列坐标，y=行坐标 形状：[H, W]
        y, x = torch.meshgrid(harange, warange, indexing='ij')

        # ==================== 核心：像素坐标 → 相机空间坐标 ====================
        # X坐标归一化：移到图片中心 / 除以焦距 → 相机坐标系X
        self.transforms_x = (x - self.width / 2) / self.focal
        # Y坐标归一化：移到图片中心 / 除以焦距 → 相机坐标系Y
        self.transforms_y = (y - self.height / 2) / self.focal

        # 生成中心裁剪索引：训练初期只采样中心区域
        self.precrop_index = torch.arange(self.width * self.height).view(self.height, self.width)
        # 计算中心区域半高/半宽
        dH = int(self.height // 2 * self.precrop_frac)
        dW = int(self.width // 2 * self.precrop_frac)
        # 切片提取中心区域索引 → 展平为一维
        self.precrop_index = self.precrop_index[
            self.height // 2 - dH: self.height // 2 + dH,
            self.width // 2 - dW: self.width // 2 + dW
        ].reshape(-1)

        # 将numpy位姿转为PyTorch张量，迁移到GPU
        poses = torch.tensor(self.poses, dtype=torch.float32, device=self.device)

        # 初始化列表：存储所有图片的光线方向+起点
        all_ray_dirs,all_ray_origins = [],[]

        # 遍历每一张图片，生成对应所有像素的光线
        for i in range(len(self.images)):
            # 调用make_rays：生成单张图片的所有光线
            ray_dirs, ray_origins = self.make_rays(self.transforms_x, self.transforms_y, poses[i])
            all_ray_dirs.append(ray_dirs)
            all_ray_origins.append(ray_origins)

        # 堆叠所有光线：[总图片数, 总像素数, 3]
        self.all_ray_dirs = torch.stack(all_ray_dirs, dim=0)
        self.all_ray_origins = torch.stack(all_ray_origins, dim=0)

        # 图片重塑：[N, H, W, 3] → [N, H*W, 3] 和光线一一对应
        self.images = torch.tensor(self.images, dtype=torch.float32, device=self.device).view(self.num_image, -1, 3)
        # 老版本错误写法（千万别用！报错神器）
        # self.images = torch.FloatTensor(self.images, device=self.device).view(self.num_image, -1, 3)

    # 核心函数：根据像素坐标+相机位姿 → 生成光线（起点+方向）
    def make_rays(self,x,y,pose):
        # x,y：归一化相机坐标
        # pose：4x4相机位姿矩阵
        # 构造相机坐标系光线方向：(x, -y, -1) 负号适配坐标系定义
        directions =torch.stack([x,-y,-torch.ones_like(x)],dim=-1)
        # 坐标系说明：
        # 屏幕坐标系：X右 Y下
        # 相机坐标系：X右 Y上 Z朝后

        # 提取相机位姿的旋转矩阵：前3行3列
        camera_matrix=pose[:3,:3]
        # 相机坐标系 → 世界坐标系：矩阵乘法 + 展平为[H*W, 3]
        ray_dirs = directions.reshape(-1,3)@ camera_matrix.T
        # 提取相机位置（光线起点）：前3行第4列 → 复制为所有光线的起点
        ray_origins = pose[:3,3].view(1,3).repeat(len(ray_dirs),1)
        return ray_dirs,ray_origins

    # 数据集长度：图片数量
    def __len__(self):
        return self.num_image

    # 采样单条数据：返回一批光线+对应像素
    def __getitem__(self, index):
        # 迭代次数+1，控制中心裁剪策略
        self.niter += 1

        # 取出当前图片的所有光线+像素
        ray_dirs = self.all_ray_dirs[index]
        ray_oris = self.all_ray_origins[index]
        img_pixels = self.images[index]

        # 前500轮：只采样中心区域光线，加速收敛
        if self.niter < self.precrop_iters:
            ray_dirs = ray_dirs[self.precrop_index]
            ray_oris = ray_oris[self.precrop_index]
            img_pixels = img_pixels[self.precrop_index]

        # 随机采样batch_size条光线，不放回采样
        nrays = self.batch_size
        select_inds = np.random.choice(ray_dirs.shape[0], size=[nrays], replace=False)
        ray_dirs = ray_dirs[select_inds]
        ray_oris = ray_oris[select_inds]
        img_pixels = img_pixels[select_inds]
        # 返回：光线方向、光线起点、对应真实像素
        return ray_dirs, ray_oris, img_pixels

    # 生成360°环绕视角的光线：用于渲染旋转视频
    def get_rotate_360_rays(self):
        # 平移变换矩阵：沿Z轴平移t
        def trans_t(t):
            return np.array([
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,t],
                [0,0,0,1],
            ], dtype=np.float32)
        # 绕X轴旋转矩阵
        def rot_phi(phi):
            return np.array([
                [1,0,0,0],
                [0,np.cos(phi),-np.sin(phi),0],
                [0,np.sin(phi), np.cos(phi),0],
                [0,0,0,1],
            ], dtype=np.float32)
        # 绕Y轴旋转矩阵
        def rot_theta(th) :
            return np.array([
                [np.cos(th),0,-np.sin(th),0],
                [0,1,0,0],
                [np.sin(th),0, np.cos(th),0],
                [0,0,0,1],
            ], dtype=np.float32)
        # 生成球面相机位姿：环绕物体360°
        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi/180.*np.pi) @ c2w
            c2w = rot_theta(theta/180.*np.pi) @ c2w
            c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
            return c2w

        # 生成41个环绕视角
        for th in np.linspace(-180., 180., 41, endpoint=False):
            pose = torch.cuda.FloatTensor(pose_spherical(th, -30., 4.), device=self.device)
            # 生成器函数：分批次生成光线，避免显存溢出
            def genfunc():
                # 修复原代码BUG：transformed_x → transforms_x
                ray_dirs, ray_origins = self.make_rays(self.transforms_x, self.transforms_y, pose)
                # 每1024条光线为一组
                for i in range(0, len(ray_dirs), 1024):
                    yield ray_dirs[i:i+1024], ray_origins[i:i+1024]
            yield genfunc

# ===================================================================================
# 360°视频渲染函数：加载训练好的模型 → 渲染环绕视频
# ===================================================================================
def make_video360(model, fine, trainset, sample_z_vals, num_importance, white_background, ckpt_path):
    """
    生成360°环绕旋转视频的函数
    参数说明：
    model:         粗网络（coarse NeRF）
    fine:          细网络（fine NeRF）
    trainset:      NeRFDataset实例，用来生成360°视角的光线
    sample_z_vals: 粗采样的深度值（和训练时保持一致）
    num_importance:细采样点数（和训练时的importance参数一致）
    white_background:是否使用白色背景
    ckpt_path:     模型权重文件路径（如 ckpt/100000.pth）
    """
    # -------------------------- 1. 加载模型权重 --------------------------
    # 加载训练好的粗网络+细网络权重
    # map_location="cuda" 直接加载到GPU，避免CPU-GPU数据搬运
    mstate, fstate = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(mstate)
    fine.load_state_dict(fstate)

    # 切换为评估模式：关闭Dropout、BatchNorm等训练专用层，不计算梯度
    model.eval()
    fine.eval()

    # 从trainset里取图片宽高，避免硬编码
    height, width = trainset.height, trainset.width
    imagelist = []  # 存储所有渲染好的帧

    # 创建保存单帧图片和视频的文件夹，避免路径不存在报错
    os.makedirs("rotate360", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    # -------------------------- 2. 遍历360°视角，逐帧渲染 --------------------------
    # trainset.get_rotate_360_rays() 会生成41个环绕视角的光线生成器
    for i, gfn in tqdm(enumerate(trainset.get_rotate_360_rays()), desc="Rendering 360° Video"):
        # 关闭梯度计算，节省显存+加速渲染
        with torch.no_grad():
            rgbs = []
            # 分批次渲染光线，避免一次性加载所有像素导致显存爆炸
            for raydirs, rayoris in gfn():
                # ===================== 🔥 修复：正确生成每帧的采样点 =====================
                sample_z_vals = torch.linspace(2.0, 6.0, 64, device='cuda').expand(len(rayoris), 64)

                # 调用render_rays渲染当前批次的光线
                # 注意：这里参数顺序必须和你定义的render_rays完全一致
                rgb_coarse, rgb_fine = render_rays(
                    coarse_net=model,
                    fine_net=fine,
                    raydirs=raydirs,
                    rayoris=rayoris,
                    z_vals=sample_z_vals,
                    num_samples2=num_importance,
                    white_background=white_background
                )
                # 我们只保留细网络的高质量渲染结果
                rgbs.append(rgb_fine)

            # 把当前帧所有批次的颜色拼接起来，恢复成完整的图片像素
            rgb = torch.cat(rgbs, dim=0)

        # -------------------------- 3. 处理渲染结果，保存单帧 --------------------------
        # 重塑为图片形状：[H*W, 3] → [H, W, 3]
        # 从GPU转到CPU，再转为numpy数组，乘以255并转为uint8（0-255像素值）
        rgb_np = (rgb.view(height, width, 3).cpu().numpy() * 255).astype(np.uint8)

        # OpenCV默认BGR格式，而渲染出来的是RGB，所以用[..., ::-1]反转通道
        file_path = f"rotate360/{i:03d}.png"
        cv2.imwrite(file_path, rgb_np[..., ::-1])
        imagelist.append(rgb_np)

    # -------------------------- 4. 合成最终视频 --------------------------
    video_path = "videos/rotate360.mp4"
    print(f"Writing video to {video_path}")
    # fps=30，和41帧匹配的话视频时长大约1.3秒，可根据需要调整
    imageio.mimwrite(video_path, imagelist, fps=30, quality=10)
    print("Video render complete!")

# ===================================================================================
# 光线采样函数：在一条光线上按深度采样3D空间点
# 公式：光线点 = 起点 + 方向 × 深度
# ===================================================================================
def sample_rays(ray_diretions, ray_origins, sample_z_vals):
    # 输入：光线方向、起点、深度值
    # 输出：光线上的所有3D采样点 + 对应深度
    nrays = len(ray_origins)
    # 深度值广播：[N_samples] → [N_rays, N_samples]
    sample_z_vals = sample_z_vals.repeat(nrays, 1)
    # 核心公式：计算光线上所有3D采样点
    rays = ray_origins[:, None, :] + ray_diretions[:, None, :] * sample_z_vals[..., None]
    return rays, sample_z_vals

# ===================================================================================
# 视角归一化函数：将光线方向转为单位向量（长度=1）
# ===================================================================================
def sample_viewdirs(ray_directions):
    return ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

# ===================================================================================
# NeRF 体渲染核心函数：密度+颜色 → 最终像素颜色
# 实现体积光渲染积分公式
# ===================================================================================
def predict_to_rgb(sigma, rgb, z_vals, raydirs, white_background=False):
    # 输入：密度、颜色、深度、光线方向、是否白色背景
    # 输出：渲染颜色、深度、累积不透明度、采样点权重
    device = sigma.device
    # 1. 计算相邻采样点的距离delta
    delta_prefix = z_vals[..., 1:] - z_vals[..., :-1]
    delta_addition = torch.full((z_vals.size(0), 1), 1e10, device=device)
    delta = torch.cat([delta_prefix, delta_addition], dim=-1)
    # 转为真实3D空间距离
    delta = delta * torch.norm(raydirs[..., None, :], dim=-1)

    # 2. 计算不透明度alpha
    alpha = 1.0 - torch.exp(-sigma * delta)

    # 3. 计算光线透射率
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    exp_addition = torch.ones((exp_term.size(0), 1), device=device)
    exp_term = torch.cat([exp_addition, exp_term + epsilon], dim=-1)
    transmittance = torch.cumprod(exp_term, axis=-1)[..., :-1]

    # 4. 计算每个采样点的贡献权重
    weights = alpha * transmittance

    # 5. 积分计算最终颜色、深度、不透明度
    rgb = torch.sum(weights[..., None] * rgb, dim=-2)
    depth = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, -1)

    # 白色背景：空白区域填充白色
    if white_background:
        rgb = rgb + (1.0 - acc_map[..., None])

    return rgb, depth, acc_map, weights

# ===================================================================================
# PDF重要性采样：根据粗网络权重，在高密度区域精细化采样（提升渲染质量）
# ===================================================================================
def sample_pdf(bins, weights, N_samples, det=True):
    device = weights.device
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if det:
        u = torch.linspace(0, 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # [N_rays, N_samples, 2]

    # --- 修复核心：调整扩展维度逻辑 ---
    # matched_shape 应该匹配 [光线数, 采样数, CDF长度]
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]

    # 提取对应的 CDF 值
    cdf_expanded = cdf.unsqueeze(1).expand(matched_shape)
    cdf_g = torch.gather(cdf_expanded, 2, inds_g)

    # 提取对应的 bins 坐标（这里 bins 也要正确扩展）
    bins_expanded = bins.unsqueeze(1).expand(matched_shape)
    bins_g = torch.gather(bins_expanded, 2, inds_g)
    # --------------------------------

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

# ===================================================================================
# 光线渲染总函数：粗采样+重要性细采样+双网络渲染
# 输入：粗网络、细网络、光线、采样参数
# 输出：粗渲染结果、最终高清渲染结果
# ===================================================================================
def render_rays(
    coarse_net,
    fine_net,
    raydirs,
    rayoris,
    z_vals,        # 粗采样深度 [N_rays, 64]
    num_samples2,  # 细采样点数 (128)
    white_background=False
):
    # 1. 视角方向归一化 (保持 [N_rays, 3])
    view_dirs_raw = sample_viewdirs(raydirs)

    # 2. 生成粗采样 3D 空间点坐标 [N_rays, 64, 3]
    rays_coarse = rayoris[..., None, :] + raydirs[..., None, :] * z_vals[..., :, None]

    # --- 修改 A：为粗网络动态扩展视角 ---
    # 使其维度与采样点一致：[N_rays, 64, 3]
    view_dirs_coarse = view_dirs_raw[:, None].repeat(1, rays_coarse.shape[1], 1)

    # 3. 粗网络预测：得到每个采样点的 (RGB, 密度)
    rgb_coarse_pts, sigma_coarse = coarse_net(rays_coarse, view_dirs_coarse)
    sigma_coarse = sigma_coarse.squeeze(dim=-1)

    # 执行体渲染积分 (Volume Integration)
    # 将 64 个点的预测值压缩成 1 条光线的颜色 [N_rays, 3]
    rgb_coarse_rendered, _, _, weights = predict_to_rgb(
        sigma_coarse,
        rgb_coarse_pts,
        z_vals,
        raydirs,
        white_background
    )

    # 5. PDF 重要性细采样 (基于粗网络的权重 weights)
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], num_samples2, det=True)
    z_samples = z_samples.detach() # 采样过程不需要梯度

    # 6. 合并粗采样(64) + 细采样(128) 的深度值并排序 → [N_rays, 192]
    z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)

    # 7. 生成细采样 3D 空间点坐标 [N_rays, 192, 3]
    rays_fine = rayoris[..., None, :] + raydirs[..., None, :] * z_vals_fine[..., :, None]

    # --- 修改 B：为细网络重新动态扩展视角 ---
    # 此时点数变多了，视角也要重新匹配：[N_rays, 192, 3]
    view_dirs_fine = view_dirs_raw[:, None].repeat(1, rays_fine.shape[1], 1)

    # 8. 细网络预测：得到 192 个点的 (RGB, 密度)
    rgb_fine_pts, sigma_fine = fine_net(rays_fine, view_dirs_fine)
    sigma_fine = sigma_fine.squeeze(dim=-1)

    #执行细网络的体渲染积分
    # 得到最终高质量渲染颜色 [N_rays, 3]
    rgb_fine_rendered, _, _, _ = predict_to_rgb(
        sigma_fine,
        rgb_fine_pts,
        z_vals_fine,
        raydirs,
        white_background
    )

    # 最终返回的是已经“积分完成”的像素颜色，这样在 train() 里做 Loss 时维度就全是 [1024, 3]
    return rgb_coarse_rendered, rgb_fine_rendered
# -----------------------------------------------------------------------------
# NeRF 核心训练函数
# 功能：循环读取光线 → 网络渲染 → 计算误差 → 更新模型 → 保存图片和权重
# -----------------------------------------------------------------------------
def train():
    # 初始化进度条：总共训练 maxiters 轮，实时显示训练状态
    pbar = tqdm(range(1, maxiters))

    # 开始逐轮训练（每一轮都会更新一次模型）
    for global_step in pbar:
        # -------------------------- 1. 随机取一张训练图 --------------------------
        # 从数据集中随机选一张图片的索引
        idx = np.random.randint(0, len(trainset))
        # 取出这张图对应的：光线方向、光线起点、真实像素颜色
        raydirs, rayoris, imagepixels = trainset[idx]

        # -------------------------- 2. 生成深度采样点（必须每轮重新生成） --------------------------
        # 在深度 2~6 之间均匀采样 num_samples1 个点（NeRF默认近远平面）
        # expand：把采样点扩展到和每一条光线对应
        sample_z_vals = torch.linspace(2.0, 6.0, num_samples1, device=device).expand(rayoris.shape[0], num_samples1)

        # -------------------------- 3. 调用渲染函数：粗网络 + 细网络 联合渲染 --------------------------
        # rgb1：粗网络渲染结果（快速、粗略）
        # rgb2：细网络渲染结果（清晰、最终输出）
        rgb1, rgb2 = render_rays(
            coarse,  # 粗网络模型
            fine,  # 细网络模型
            raydirs,  # 光线方向
            rayoris,  # 光线起点
            sample_z_vals,  # 粗采样深度
            num_samples2,  # 细采样点数（重要性采样）
            white_background  # 是否使用白色背景
        )

        # -------------------------- 4. 计算损失（误差） --------------------------
        # 粗网络损失：渲染结果 和 真实图片 的均方误差
        loss1 = ((rgb1 - imagepixels) ** 2).mean()
        # 细网络损失：最终渲染结果 和 真实图片 的均方误差（主要监督信号）
        loss2 = ((rgb2 - imagepixels) ** 2).mean()
        # 总损失：粗+细一起训练，让两个网络都学好
        loss = loss1 + loss2

        # -------------------------- 5. 计算 PSNR（图像质量指标） --------------------------
        # PSNR 越高 → 渲染越清晰、越接近真实图
        psnr = 10. * torch.log10(1.0 / loss2.detach())

        # -------------------------- 6. 反向传播 → 更新网络参数（核心学习步骤） --------------------------
        optimizer.zero_grad()  # 清空上一轮的梯度，避免叠加干扰
        loss.backward()  # 反向传播：计算每个参数需要更新的方向
        optimizer.step()  # 更新参数：让网络往误差更小的方向走一步

        # -------------------------- 7. 更新进度条显示信息 --------------------------
        pbar.set_description(
            f"step:{global_step}/{maxiters} loss:{loss.item():.4f} psnr:{psnr.item():.2f}"
        )

        # -------------------------- 8. 学习率衰减（让模型后期学得更精细） --------------------------
        # 学习率随训练轮数逐渐变小，前期大步学，后期精细调
        decay_rate = 0.1  # 衰减系数
        new_lrate = lrate * (decay_rate ** (global_step / lrate_decay))
        # 把新学习率设置到优化器里
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # -------------------------- 9. 定期保存测试图片 + 模型权重 --------------------------
        # 每 5000 轮 或 第 500 轮，保存一次结果
        if global_step % 5000 == 0 or global_step == 500:
            # 图片保存路径 + 模型保存路径
            imgpath = f"imgs/{global_step:02d}.png"
            pthpath = f"ckpt/{global_step:02d}.pth"

            # 自动创建保存文件夹，不存在就新建
            os.makedirs("imgs", exist_ok=True)
            os.makedirs("ckpt", exist_ok=True)

            # -------------------- 测试阶段：切换为评估模式（不计算梯度） --------------------
            coarse.eval()  # 粗网络 → 评估模式
            fine.eval()  # 细网络 → 评估模式
            with torch.no_grad():  # 关闭梯度，节省显存、加速计算
                # 取第0张图作为测试图，渲染一张完整图片
                test_idx = 0
                raydirs_test, rayoris_test, imagepixels_test = trainset[test_idx]
                # 生成测试用深度采样
                sample_z_vals_test = torch.linspace(2.0, 6.0, num_samples1, device=device).expand(rayoris_test.shape[0],
                                                                                                  num_samples1)
                # 渲染测试图
                rgb1, rgb2 = render_rays(coarse, fine, raydirs_test, rayoris_test, sample_z_vals_test, num_samples2,
                                         white_background)

                # 计算测试集损失和PSNR
                test_loss = ((rgb2 - imagepixels_test) ** 2).mean()
                test_psnr = 10. * torch.log10(1.0 / test_loss.detach())

                # 打印保存信息
                print(f"Save {imgpath} | test_loss: {test_loss.item():.6f} | test_psnr: {test_psnr.item():.2f}")

                # -------------------- 把渲染结果转成图片并保存 --------------------
                # 把网络输出的一维像素，恢复成 高×宽×3通道 的图片形状
                temp_image = (rgb2.view(provider.height, provider.width, 3).cpu().numpy() * 255).astype(np.uint8)
                # OpenCV 用 BGR 格式，网络输出是 RGB，要翻转通道才能保存正确颜色
                cv2.imwrite(imgpath, temp_image[..., ::-1])

                # -------------------- 保存模型权重（以后可加载继续训练/渲染视频） --------------------
                torch.save([coarse.state_dict(), fine.state_dict()], pthpath)

            # 测试完成，切回训练模式，继续下一轮训练
            coarse.train()
            fine.train()


# -----------------------------------------------------------------------------
# 主函数：程序入口
# 功能：配置参数 → 加载数据集 → 创建网络 → 启动训练
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # ===================== 一、基础配置参数 =====================
    root = "data/nerf_synthetic/lego"  # 数据集路径（lego数据集）
    transforms_file = "transforms_train.json"  # 相机参数文件
    half_resolution = False  # 是否减半分辨率（False=高清，True=更快）
    batch_size = 1024  # 每轮训练多少条光线
    device = "cuda"  # 用GPU训练

    # ===================== 二、加载数据集 =====================
    # 读取数据集：图片、相机位姿、焦距、图片宽高等全部准备好
    provider = DatabaseProvider(root, transforms_file, half_resolution)
    # 把数据集包装成NeRF专用格式：自动生成所有光线
    trainset = NeRFDataset(provider, batch_size, device)

    # ===================== 三、创建NeRF网络（粗+细两个模型） =====================
    x_pedim = 10  # 位置编码层数（越高细节越多）
    view_pedim = 4  # 视角编码层数（越高反光越真实）
    # 创建粗网络：先粗略采样
    coarse = NeRF(x_pedim=x_pedim, view_pedim=view_pedim).to(device)
    # 创建细网络：在粗网络基础上精细采样
    fine = NeRF(x_pedim=x_pedim, view_pedim=view_pedim).to(device)
    # 把粗+细网络的参数合并，一起训练
    params = list(coarse.parameters()) + list(fine.parameters())

    # ===================== 四、优化器设置（让网络学习的工具） =====================
    lrate = 5e-4  # 学习率（越大学习越快，但容易不稳）
    lrate_decay = 500 * 1000  # 学习率衰减步数
    # Adam优化器：深度学习最常用、最稳定的优化器
    optimizer = optim.Adam(params, lr=lrate)

    # ===================== 五、渲染核心参数 =====================
    num_samples1 = 64  # 粗网络每条光线采样64个点
    num_samples2 = 128  # 细网络额外再采样128个点
    white_background = True  # 白色背景（lego数据集标准设置）

    # ===================== 六、训练总轮数 =====================
    maxiters = 100000 + 1  # 总共训练10万轮

    # ===================== 七、预热检查：先跑一帧确保无报错 =====================
    ray_dirs, ray_oris, image_pixels = trainset[0]
    sample_z_vals = torch.linspace(2.0, 6.0, num_samples1, device=device).expand(ray_oris.shape[0], num_samples1)

    # 启动训练！
    print("开始训练！")
    train()
    make_video360(
        model=coarse,
        fine=fine,
        trainset=trainset,
        sample_z_vals=sample_z_vals,
        num_importance=num_samples2,
        white_background=True,
        ckpt_path="ckpt/300000.pth"  # 改成你实际保存的模型路径
    )
