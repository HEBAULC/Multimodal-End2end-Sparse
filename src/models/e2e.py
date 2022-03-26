import torch
from torch import nn
from src.models.e2e_t import MME2E_T

from src.models.transformer_encoder import WrappedTransformerEncoder
from torchvision import transforms
from facenet_pytorch import MTCNN
from src.models.vgg_block import VggBasicBlock


class MME2E(nn.Module):
	# 魔法函数
    def __init__(self, args, device):
        super(MME2E, self).__init__()
        # 情感类别数量
        self.num_classes = args['num_emotions']
        # 超参数
        self.args = args
        self.mod = args['modalities'].lower()
        self.device = device
        # 特征维度
        self.feature_dim = args['feature_dim']
        # 训练的层数
        nlayers = args['trans_nlayers']
        # 训练的头数
        nheads = args['trans_nheads']
        # 训练的维度
        trans_dim = args['trans_dim']
		
		# 以下都是进入transformer之前的准备工作 也就是模型自动提取特征的部分
		
		#文本嵌入维度
        # base
        text_cls_dim = 768

        #文本模型的大小
        if args['text_model_size'] == 'large':
            text_cls_dim = 1024
        if args['text_model_size'] == 'xlarge':
            text_cls_dim = 2048
		
		# 文本模态调用MME2E_T函数
        self.T = MME2E_T(feature_dim=self.feature_dim, size=args['text_model_size'])
		
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
		
		# 将变换后视频和音频特征送入 包裹式transformer编码器
        self.v_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)
        self.a_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)

        # 对视频的FFN 全连接层 trans_dim->num_classes
        self.v_out = nn.Linear(trans_dim, self.num_classes)
        # 对文本的的FFN 全连接层 text_cls_dim->num_classes
        self.t_out = nn.Linear(text_cls_dim, self.num_classes)
        # 对音频的的FFN 全连接层 trans_dim->num_classes
        self.a_out = nn.Linear(trans_dim, self.num_classes)
        # 加权融合 默认情况下 3->1
        self.weighted_fusion = nn.Linear(len(self.mod), 1, bias=False)

	# 前向传播
    def forward(self, imgs, imgs_lens, specs, spec_lens, text):
        all_logits = []

        if 't' in self.mod:
            text_cls = self.T(text, get_cls=True)
            # FFN text_feature(768)将仿射变幻后的feature_dim(512)维度映射为num_classes个输出
            text_cls = self.t_out(text_cls)
            # print('text_cls', text_cls)
            '''
            text_cls tensor([[-0.6526,  0.1766, -0.0253,  0.1428,  0.1186,  0.0980],
                            [-0.6833,  0.2985, -0.0351,  0.2437,  0.0967,  0.0778],
                            [-0.6604,  0.2114, -0.0279,  0.1628,  0.1273,  0.0867],
                            [-0.6558,  0.2613, -0.0494,  0.1803,  0.1252,  0.0957],
                            [-0.6566,  0.2206, -0.0264,  0.1623,  0.1226,  0.0842],
                            [-0.6450,  0.1931, -0.0131,  0.1729,  0.1084,  0.0509],
                            [-0.6558,  0.3035,  0.0040,  0.2145,  0.0763,  0.0873],
                            [-0.6632,  0.2482, -0.0475,  0.1790,  0.1339,  0.0954]],
                            device='cuda:0')
            '''
            # print('text_cls.size()', text_cls.size()) #torch.Size([8, 6])
            all_logits.append(text_cls)
            # print('all_logits_1', all_logits)
            '''
            all_logits_1 [tensor([[-0.6526,  0.1766, -0.0253,  0.1428,  0.1186,  0.0980],
                                [-0.6833,  0.2985, -0.0351,  0.2437,  0.0967,  0.0778],
                                [-0.6604,  0.2114, -0.0279,  0.1628,  0.1273,  0.0867],
                                [-0.6558,  0.2613, -0.0494,  0.1803,  0.1252,  0.0957],
                                [-0.6566,  0.2206, -0.0264,  0.1623,  0.1226,  0.0842],
                                [-0.6450,  0.1931, -0.0131,  0.1729,  0.1084,  0.0509],
                                [-0.6558,  0.3035,  0.0040,  0.2145,  0.0763,  0.0873],
                                [-0.6632,  0.2482, -0.0475,  0.1790,  0.1339,  0.0954]],
                        device='cuda:0')]
            '''
            # all_logits.append(self.t_out(text_cls))

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
            # FFN 将transformer Encoder的输出维度映射为num_classes个输出
            faces = self.v_out(faces)
            # print('faces.size()', faces.size()) # torch.Size([8, 6])
            # print(faces)
            '''
            tensor([[ 1.5943e+00, -1.6899e+00,  1.5271e-01, -9.2948e-01, -3.4113e+00,  -8.3180e-01],
                [ 1.2225e+00, -1.2569e+00,  2.6603e-01, -5.9163e-01,  2.0862e-01, -3.2843e+00],
                [ 1.1931e+00, -1.2046e+00,  2.3207e-01, -5.3862e-01,  3.2041e-01, -3.3370e+00],
                [ 1.7229e+00, -1.7335e+00,  2.7590e-01, -7.9337e-01, -1.7266e+00,  -2.6737e+00],
                [ 1.1501e+00, -1.2031e+00,  2.2313e-01, -5.1021e-01,  4.1232e-01, -3.3517e+00],
                [ 1.1898e+00, -1.5256e+00, -1.9559e-03, -8.9082e-01, -3.7326e+00, 2.8615e-01],
                [ 1.1994e+00, -1.2865e+00,  2.7312e-01, -5.9162e-01,  2.2658e-01, -3.2646e+00],
                [ 1.1619e+00, -1.4713e+00, -6.2906e-02, -8.8829e-01, -3.7255e+00, 3.2427e-01]], device='cuda:0')]
            '''
            all_logits.append(faces)
            # print('all_logits_2', all_logits)
            '''
            all_logits_2 [tensor([[-0.6526,  0.1766, -0.0253,  0.1428,  0.1186,  0.0980],
                                    [-0.6833,  0.2985, -0.0351,  0.2437,  0.0967,  0.0778],
                                    [-0.6604,  0.2114, -0.0279,  0.1628,  0.1273,  0.0867],
                                    [-0.6558,  0.2613, -0.0494,  0.1803,  0.1252,  0.0957],
                                    [-0.6566,  0.2206, -0.0264,  0.1623,  0.1226,  0.0842],
                                    [-0.6450,  0.1931, -0.0131,  0.1729,  0.1084,  0.0509],
                                    [-0.6558,  0.3035,  0.0040,  0.2145,  0.0763,  0.0873],
                                    [-0.6632,  0.2482, -0.0475,  0.1790,  0.1339,  0.0954]],
                        device='cuda:0'), 
                           tensor([[ 1.5943e+00, -1.6899e+00,  1.5271e-01, -9.2948e-01, -3.4113e+00,  -8.3180e-01],
                                    [ 1.2225e+00, -1.2569e+00,  2.6603e-01, -5.9163e-01,  2.0862e-01, -3.2843e+00],
                                    [ 1.1931e+00, -1.2046e+00,  2.3207e-01, -5.3862e-01,  3.2041e-01, -3.3370e+00],
                                    [ 1.7229e+00, -1.7335e+00,  2.7590e-01, -7.9337e-01, -1.7266e+00,  -2.6737e+00],
                                    [ 1.1501e+00, -1.2031e+00,  2.2313e-01, -5.1021e-01,  4.1232e-01, -3.3517e+00],
                                    [ 1.1898e+00, -1.5256e+00, -1.9559e-03, -8.9082e-01, -3.7326e+00, 2.8615e-01],
                                    [ 1.1994e+00, -1.2865e+00,  2.7312e-01, -5.9162e-01,  2.2658e-01, -3.2646e+00],
                                    [ 1.1619e+00, -1.4713e+00, -6.2906e-02, -8.8829e-01, -3.7255e+00, 3.2427e-01]], device='cuda:0')]
            '''

            # all_logits.append(self.v_out(faces))

        if 'a' in self.mod:
            for a_module in self.A:
                specs = a_module(specs)

            specs = self.a_flatten(specs.flatten(start_dim=1))
            specs = self.a_transformer(specs, spec_lens, get_cls=True)
            # FFN 将transformer Encoder的输出维度映射为num_classes个输出
            specs = self.a_out(specs)
            # print('specs.size()', specs.size()) # torch.Size([8, 6])
            # print(specs)
            '''
            tensor([[-2.8051, -1.2063, -1.4808,  0.6384, -0.0477,  4.4180],
                    [ 3.2900,  2.4277,  0.8467, -1.0573, -0.0261, -3.6165],
                    [ 3.3344,  2.4606,  0.8881, -1.0380, -0.1907, -3.4880],
                    [ 3.5769,  2.4294,  0.5269, -0.8636, -1.5080, -1.6498],
                    [ 3.3518,  2.4772,  0.9140, -1.0232, -0.3048, -3.3914],
                    [-3.4560, -1.3998, -1.1127,  0.7102,  0.5030,  3.9598],
                    [ 3.2739,  2.4163,  0.8332, -1.0633,  0.0209, -3.6495],
                    [ 3.3334,  2.4605,  0.8882, -1.0381, -0.1901, -3.4887]],
                device='cuda:0')]
            '''
            all_logits.append(specs)
            # print('all_logits_3', all_logits)
            '''
            all_logits_3 [tensor([[-0.6526,  0.1766, -0.0253,  0.1428,  0.1186,  0.0980],
                                    [-0.6833,  0.2985, -0.0351,  0.2437,  0.0967,  0.0778],
                                    [-0.6604,  0.2114, -0.0279,  0.1628,  0.1273,  0.0867],
                                    [-0.6558,  0.2613, -0.0494,  0.1803,  0.1252,  0.0957],
                                    [-0.6566,  0.2206, -0.0264,  0.1623,  0.1226,  0.0842],
                                    [-0.6450,  0.1931, -0.0131,  0.1729,  0.1084,  0.0509],
                                    [-0.6558,  0.3035,  0.0040,  0.2145,  0.0763,  0.0873],
                                    [-0.6632,  0.2482, -0.0475,  0.1790,  0.1339,  0.0954]],
                        device='cuda:0'),
                        tensor([[ 1.5943e+00, -1.6899e+00,  1.5271e-01, -9.2948e-01, -3.4113e+00, -8.3180e-01],
                                [ 1.2225e+00, -1.2569e+00,  2.6603e-01, -5.9163e-01,  2.0862e-01, -3.2843e+00],
                                [ 1.1931e+00, -1.2046e+00,  2.3207e-01, -5.3862e-01,  3.2041e-01, -3.3370e+00],
                                [ 1.7229e+00, -1.7335e+00,  2.7590e-01, -7.9337e-01, -1.7266e+00, -2.6737e+00],
                                [ 1.1501e+00, -1.2031e+00,  2.2313e-01, -5.1021e-01,  4.1232e-01, -3.3517e+00],
                                [ 1.1898e+00, -1.5256e+00, -1.9559e-03, -8.9082e-01, -3.7326e+00, 2.8615e-01],
                                [ 1.1994e+00, -1.2865e+00,  2.7312e-01, -5.9162e-01,  2.2658e-01, -3.2646e+00],
                                [ 1.1619e+00, -1.4713e+00, -6.2906e-02, -8.8829e-01, -3.7255e+00, 3.2427e-01]], device='cuda:0'),
                       tensor([[-2.8051, -1.2063, -1.4808,  0.6384, -0.0477,  4.4180],
                                [ 3.2900,  2.4277,  0.8467, -1.0573, -0.0261, -3.6165],
                                [ 3.3344,  2.4606,  0.8881, -1.0380, -0.1907, -3.4880],
                                [ 3.5769,  2.4294,  0.5269, -0.8636, -1.5080, -1.6498],
                                [ 3.3518,  2.4772,  0.9140, -1.0232, -0.3048, -3.3914],
                                [-3.4560, -1.3998, -1.1127,  0.7102,  0.5030,  3.9598],
                                [ 3.2739,  2.4163,  0.8332, -1.0633,  0.0209, -3.6495],
                                [ 3.3334,  2.4605,  0.8882, -1.0381, -0.1901, -3.4887]],
                            device='cuda:0')]
            '''
            # all_logits.append(self.a_out(specs))

        # 如果只有一个模态
        if len(self.mod) == 1:
            return all_logits[0]

        # torch.stack(all_logits, dim=-1) [8, 18=6*3]
        stack = torch.stack(all_logits, dim=-1)
        # print(stack.size()) # torch.Size([8, 6 ,3])
        # print(stack)
        '''
        tensor([[[-6.5263e-01,  1.5943e+00, -2.8051e+00],
                 [ 1.7661e-01, -1.6899e+00, -1.2063e+00],
                 [-2.5304e-02,  1.5271e-01, -1.4808e+00],
                 [ 1.4278e-01, -9.2948e-01,  6.3838e-01],
                 [ 1.1863e-01, -3.4113e+00, -4.7680e-02],
                 [ 9.7985e-02, -8.3180e-01,  4.4180e+00]],
                [[-6.8334e-01,  1.2225e+00,  3.2900e+00],
                 [ 2.9852e-01, -1.2569e+00,  2.4277e+00],
                 [-3.5117e-02,  2.6603e-01,  8.4666e-01],
                 [ 2.4365e-01, -5.9163e-01, -1.0573e+00],
                 [ 9.6652e-02,  2.0862e-01, -2.6145e-02],
                 [ 7.7764e-02, -3.2843e+00, -3.6165e+00]],
                [[-6.6043e-01,  1.1931e+00,  3.3344e+00],
                 [ 2.1140e-01, -1.2046e+00,  2.4606e+00],
                 [-2.7884e-02,  2.3207e-01,  8.8812e-01],
                 [ 1.6275e-01, -5.3862e-01, -1.0380e+00],
                 [ 1.2729e-01,  3.2041e-01, -1.9075e-01],
                 [ 8.6736e-02, -3.3370e+00, -3.4880e+00]],
                [[-6.5577e-01,  1.7229e+00,  3.5769e+00],
                 [ 2.6132e-01, -1.7335e+00,  2.4294e+00],
                 [-4.9394e-02,  2.7590e-01,  5.2685e-01],
                 [ 1.8030e-01, -7.9337e-01, -8.6360e-01],
                 [ 1.2520e-01, -1.7266e+00, -1.5080e+00],
                 [ 9.5682e-02, -2.6737e+00, -1.6498e+00]],
                [[-6.5665e-01,  1.1501e+00,  3.3518e+00],
                 [ 2.2065e-01, -1.2031e+00,  2.4772e+00],
                 [-2.6383e-02,  2.2313e-01,  9.1399e-01],
                 [ 1.6228e-01, -5.1021e-01, -1.0232e+00],
                 [ 1.2258e-01,  4.1232e-01, -3.0483e-01],
                 [ 8.4220e-02, -3.3517e+00, -3.3914e+00]],
                [[-6.4504e-01,  1.1898e+00, -3.4560e+00],
                 [ 1.9307e-01, -1.5256e+00, -1.3998e+00],
                 [-1.3051e-02, -1.9559e-03, -1.1127e+00],
                 [ 1.7288e-01, -8.9082e-01,  7.1023e-01],
                 [ 1.0844e-01, -3.7326e+00,  5.0303e-01],
                 [ 5.0860e-02,  2.8615e-01,  3.9598e+00]],
                [[-6.5578e-01,  1.1994e+00,  3.2739e+00],
                 [ 3.0354e-01, -1.2865e+00,  2.4163e+00],
                 [ 3.9931e-03,  2.7312e-01,  8.3315e-01],
                 [ 2.1447e-01, -5.9162e-01, -1.0633e+00],
                 [ 7.6307e-02,  2.2658e-01,  2.0903e-02],
                 [ 8.7251e-02, -3.2646e+00, -3.6495e+00]],
                [[-6.6317e-01,  1.1619e+00,  3.3334e+00],
                 [ 2.4819e-01, -1.4713e+00,  2.4605e+00],
                 [-4.7499e-02, -6.2906e-02,  8.8823e-01],
                 [ 1.7901e-01, -8.8829e-01, -1.0381e+00],
                 [ 1.3392e-01, -3.7255e+00, -1.9009e-01],
                 [ 9.5392e-02,  3.2427e-01, -3.4887e+00]]], device='cuda:0')
        '''
        # self.weighted_fusion(stack) [8, 6 ,3]->[8, 6 ,1]
        # .squeeze(-1) [8, 6 ,1]->[8, 6]
        return self.weighted_fusion(stack).squeeze(-1)
	
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
        off = (current_size - target_size) // 2 # offset
        cropped = img[:, off:off + target_size, off - target_size // 2:off + target_size // 2]
        return cropped
