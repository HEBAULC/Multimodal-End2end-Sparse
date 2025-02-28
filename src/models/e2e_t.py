import torch
from torch import nn
from transformers import BertModel
from transformers import AlbertModel
from transformers import RobertaModel

class MME2E_T(nn.Module):
    def __init__(self, feature_dim, num_classes=4, size='base'):
        super(MME2E_T, self).__init__()
        self.albert = AlbertModel.from_pretrained(f'albert-{size}-v2')
        # 文本特征仿射 原来albert是注释掉的也不出错 不注释也能运行
        # self.text_feature_affine = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, feature_dim)
        # )

        # self.albert = AlbertModel.from_pretrained(f'voidful/albert_chinese_base') #base 768 large 1024
        # print(feature_dim)
        # self.text_feature_affine = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, feature_dim) #feature_dim=256
        # )

        # self.bert = BertModel.from_pretrained('bert-base-chinese')

        # self.albert = RobertaModel.from_pretrained(f'roberta-{size}')


    def forward(self, text, get_cls=False):
        # logits, hidden_states = self.albert(**text, output_hidden_states=True)
        last_hidden_state, _ = self.albert(**text)
        # print('last_hidden_state.size', last_hidden_state.size()) #torch.Size([8, 50, 768])
        # last_hidden_state, _ = self.bert(**text)

        # 获取句子特征
        if get_cls:
            # 取第一列的元素
            cls_feature = last_hidden_state[:, 0]
            # cls_feature = self.text_feature_affine(cls_feature)
            return cls_feature

        # 获取文本特征
        # text_features = self.text_feature_affine(last_hidden_state).sum(1) # 在1轴进行相加 squence_lenth所在轴
        text_features = last_hidden_state.sum(1)
        return text_features
