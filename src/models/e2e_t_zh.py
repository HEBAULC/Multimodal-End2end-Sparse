import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig


class MME2E_T_ZH(nn.Module):
    def __init__(self, feature_dim, num_classes=4, size='base'):
        super(MME2E_T_ZH, self).__init__()
        # self.text_feature_affine = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, feature_dim) #feature_dim=256
        # )

        pretrained = 'voidful/albert_chinese_base'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = BertModel.from_pretrained(pretrained)
        self.config = BertConfig.from_pretrained(pretrained)

    def forward(self, input_text, get_cls=False):
        tokenized_text = self.tokenizer.encode(input_text)
        '''
        Input {'input_ids': tensor([[]]), device='cuda:0')} is not valid. 
        Should be a string, a list/tuple of strings or a list/tuple of integers.
        '''
        input_ids = torch.tensor(tokenized_text).view(-1, len(tokenized_text))
        outputs = self.model(input_ids)
        print(outputs[0].shape, outputs[1].shape)
        # print('last_hidden_state.size', last_hidden_state.size()) #torch.Size([8, 50, 768])

        # 获取句子特征
        if get_cls:
            # 取第一列的元素
            # cls_feature = last_hidden_state[:, 0]
            cls_feature = outputs[1]
            # cls_feature = self.text_feature_affine(cls_feature)
            return cls_feature

        # 获取文本特征
        # text_features = self.text_feature_affine(last_hidden_state).sum(1) # 在1轴进行相加 squence_lenth所在轴
        text_features = outputs[0].sum(1)
        return text_features
