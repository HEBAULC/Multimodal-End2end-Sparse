import torch
from torch import nn
from transformers import AlbertModel

class MME2E_T(nn.Module):
    def __init__(self, feature_dim, num_classes=4, size='base'):
        super(MME2E_T, self).__init__()
        self.albert = AlbertModel.from_pretrained(f'albert-{size}-v2')
        # 文本特征仿射 原来是注释掉的也不出错
        self.text_feature_affine = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, text, get_cls=False):
        # logits, hidden_states = self.albert(**text, output_hidden_states=True)
        last_hidden_state, _ = self.albert(**text)

        # 获取句子特征
        if get_cls:
            # 取第一列的元素
            cls_feature = last_hidden_state[:, 0]
            # cls_feature = self.text_feature_affine(cls_feature)
            return cls_feature

        # 获取文本特征
        text_features = self.text_feature_affine(last_hidden_state).sum(1)
        return text_features
