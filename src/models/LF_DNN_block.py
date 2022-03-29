import torch
from torch import nn
import torch.nn.functional as F
# from FeatureNets import SubNet, TextSubNet


class LF_DNN_Block(nn.Module):
    """
    late fusion using DNN
    使用深度神经网络进行后期融合
    """

    def __init__(self, num_classes, dropouts):
        super(LF_DNN_Block, self).__init__()
        post_fusion_dim, output_dim = (32, num_classes)
        # 后期融合的维度
        self.text_hidden_size, self.audio_hidden_size, self.video_hidden_size = (128, 32, 128)

        self.post_fusion_dim = post_fusion_dim

        # 三模态的丢弃率 也是设置为相同
        # self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = dropouts
        self.post_fusion_prob = dropouts

        # define the pre-fusion subnetworks
        # 定义融合前的子网络 接在transformer后
        # 音频子网络 总体上看 audio_in_size -> audio_hidden_size
        # self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        #
        # # 视频子网络 总体上看 video_in_size -> video_hidden_size
        # self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        #
        # # 文本子网络 总体上看 text_in_size -> text_hidden_size
        # self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        # 定义后期融合网络 LF-DNN
        # 定义丢弃率 失活率
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)

        # 后期融合层1 (t_h+ v_h + a_h) -> post_fusion_dim
        self.post_fusion_layer_1 = nn.Linear(self.text_hidden_size + self.video_hidden_size + self.audio_hidden_size,
                                             self.post_fusion_dim)
        # 后期融合层2 post_fusion_dim -> post_fusion_dim
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        # 后期融合层3 post_fusion_dim -> output_dim
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, output_dim)

    def forward(self, audio_h, video_h, text_h):
        # 三模态隐藏层在最后一个维度上进行拼接 最后一个维度也就是特征维
        fusion_h = torch.cat([audio_h, video_h, text_h], dim=-1)
        # 进入后期融合层之前的随机失活
        x = self.post_fusion_dropout(fusion_h)

        # 后期融合层1处理 再relu激活
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)

        # 后期融合层2处理 再relu激活
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)

        # 后期融合层2处理 不使用激活函数
        output = self.post_fusion_layer_3(x)

        # 实例化一个嵌套字典
        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            # 三模态的拼接张量
            'Feature_f': fusion_h,
            'M': output
        }

        return output

        # 返回嵌套字典
        # return res
