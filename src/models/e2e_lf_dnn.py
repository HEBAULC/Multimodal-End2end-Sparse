import torch
from torch import nn
from src.models.e2e_t import MME2E_T

from src.models.transformer_encoder import WrappedTransformerEncoder
from torchvision import transforms
from facenet_pytorch import MTCNN
from src.models.vgg_block import VggBasicBlock
from src.models.LF_DNN_block import LF_DNN_Block


class MME2E_LFDNN(nn.Module):
    # 魔法函数
    def __init__(self, args, device):
        super(MME2E_LFDNN, self).__init__()
        # 情感类别数量
        self.num_classes = args['num_emotions']
        # 超参数
        self.args = args
        self.mod = args['modalities'].lower()
        self.device = device
        # 每个模态最终输出的特征维度 default=256
        self.feature_dim = args['feature_dim']

        dropout = 0.4
        text_hidden_size, video_hidden_size, audio_hidden_size = (128, 32, 128)

        # 训练的层数
        nlayers = args['trans_nlayers']
        # 训练的头数
        nheads = args['trans_nheads']
        # 训练的transformer维度 default=512
        trans_dim = args['trans_dim']

        # 以下都是进入transformer之前的准备工作 也就是模型自动提取特征的部分

        # 文本嵌入维度
        # base
        text_cls_dim = 768

        # 文本模型的大小
        if args['text_model_size'] == 'large':
            text_cls_dim = 1024
        if args['text_model_size'] == 'xlarge':
            text_cls_dim = 2048

        # 文本模态调用MME2E_T函数
        # 768->512->256
        self.T = MME2E_T(feature_dim=self.feature_dim, size=args['text_model_size'])

        # MME2E_T中有affine这里就不用affine
        self.Text_affine = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_cls_dim, self.feature_dim), #feature_dim 512
            nn.Linear(self.feature_dim, text_hidden_size) #text_hidden_size 128
        )

        # video, audio 输入transformer编码器之前的准备工作
        # 视觉模态的预处理
        # 调用MTCNN模块提取人脸 from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(image_size=48, margin=2, post_process=False, device=device)
        # 归一化
        self.normalize = transforms.Normalize(mean=[159, 111, 102], std=[37, 33, 32])

        # 处理视觉模态的特征
        self.V = nn.Sequential(
            # 2d卷积提取特征
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            # 批量归一化
            nn.BatchNorm2d(64),
            # ReLU激活 近似
            nn.ReLU(),
            # 最大池化 汇聚特征
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 调用3次VGG基本模块
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            # 最大池化 汇聚提取的特征
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 调用一次VGG基本模块 运用最大池化
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 调用一次VGG基本模块 运用最大池化
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 音频模态的模型
        # 和对图像处理的模型一摸一样，对MFCC频率图像进行特征提取
        self.A = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 对视觉模态特征进行线性变换+非线性变换 变换到transformer的输入维度
        self.v_flatten = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        # 对音频模态特征进行线性变换+非线性变换 变换到transformer的输入维度
        self.a_flatten = nn.Sequential(
            nn.Linear(512 * 8 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        # 输入transformer编码器
        # 将变换后视频和音频特征送入 包裹式transformer编码器
        self.v_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)
        self.a_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)

        # (512,32)
        self.v_hidden = nn.Linear(trans_dim, video_hidden_size)
        # (512,128)
        self.a_hidden = nn.Linear(trans_dim, audio_hidden_size)

        #self.Text_affine, self.a_hidden, self.v_hidden,
        self.weighted_fusion = LF_DNN_Block(self.num_classes, dropout)

    # 前向传播
    def forward(self, imgs, imgs_lens, specs, spec_lens, text):
        global text_hidden, audio_hidden, video_hidden
        all_logits = []

        if 't' in self.mod:
            text_cls= self.T(text, get_cls=True)
            # FFN text_feature(768)将仿射变幻后的feature_dim(256)维度映射为num_classes个输出
            # RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x768 and 256x128)

            # RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x1024 and 768x256)
            text_hidden = self.Text_affine(text_cls)

            # text_cls = self.t_out(text_cls)
            # all_logits.append(text_cls)

        if 'v' in self.mod:
            faces = self.mtcnn(imgs)
            for i, face in enumerate(faces):
                if face is None:
                    center = self.crop_img_center(torch.tensor(imgs[i]).permute(2, 0, 1))
                    faces[i] = center
            faces = [self.normalize(face) for face in faces]
            faces = torch.stack(faces, dim=0).to(device=self.device)

            faces = self.V(faces)

            faces = self.v_flatten(faces.flatten(start_dim=1))
            # 特征送入
            faces = self.v_transformer(faces, imgs_lens, get_cls=True)
            video_hidden = self.v_hidden(faces)

            # FFN 将transformer Encoder的输出维度映射为num_classes个输出
            # faces = self.v_out(faces)
            # all_logits.append(faces)

        if 'a' in self.mod:
            for a_module in self.A:
                specs = a_module(specs)

            specs = self.a_flatten(specs.flatten(start_dim=1))
            specs = self.a_transformer(specs, spec_lens, get_cls=True)
            audio_hidden = self.a_hidden(specs)

            # FFN 将transformer Encoder的输出维度映射为num_classes个输出
            # specs = self.a_out(specs)
            # all_logits.append(specs)

        # 如果只有一个模态
        if len(self.mod) == 1:
            return all_logits[0]

        # torch.stack(all_logits, dim=-1) [8, 18=6*3]
        # stack = torch.stack(all_logits, dim=-1)
        # return self.weighted_fusion(stack).squeeze(-1)

        # .squeeze()
        return self.weighted_fusion(text_hidden, audio_hidden, video_hidden)

    # 图像中心裁剪
    def crop_img_center(self, img: torch.tensor, target_size=48):
        '''
        Some images have un-detectable faces,
        to make the training goes normally,
        for those images, we crop the center part,
        which highly likely contains the face or part of the face.

        有些图像有无法检测到的人脸，
		为了让训练正常进行，
		对于这些图像，我们裁剪中心部分，
		很可能包含面部或部分面部。

        @img - (channel, height, width)
        '''
        current_size = img.size(1)
        off = (current_size - target_size) // 2  # offset
        cropped = img[:, off:off + target_size, off - target_size // 2:off + target_size // 2]
        return cropped
