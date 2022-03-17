import os
import pickle
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# 保存pickle文件
def save(toBeSaved, filename, mode='wb'):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file = open(filename, mode)
    pickle.dump(toBeSaved, file, protocol=4)
    file.close()

# 加载pickle文件
def load(filename, mode='rb'):
    file = open(filename, mode)
    loaded = pickle.load(file)
    file.close()
    return loaded

# For python2
def load2(path):
    with open(path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return p

# 填充句子
def pad_sents(sents, pad_token):
    sents_padded = []
    lens = get_lens(sents)
    max_len = max(lens)
    sents_padded = [sents[i] + [pad_token] * (max_len - l) for i, l in enumerate(lens)]
    return sents_padded, lens

# 按长度对句子进行排序
def sort_sents(sents, reverse=True):
    sents.sort(key=(lambda s: len(s)), reverse=reverse)
    return sents

# 获取掩码序列1111000
def get_mask(sents, unmask_idx=1, mask_idx=0):
    lens = get_lens(sents)
    max_len = max(lens)
    mask = [([unmask_idx] * l + [mask_idx] * (max_len - l)) for l in lens]
    return mask

# 获取句子长度
def get_lens(sents):
    return [len(sent) for sent in sents]

# 获取最大句子长度
def get_max_len(sents):
    max_len = max([len(sent) for sent in sents])
    return max_len

# 根据有效长度截取句子
def truncate_sents(sents, length):
    sents = [sent[:length] for sent in sents]
    return sents

# 获取损失权重
def get_loss_weight(labels, label_order):
    nums = [np.sum(labels == lo) for lo in label_order]
    loss_weight = torch.tensor([n / len(labels) for n in nums])
    return loss_weight

# 首字母大写
def capitalize_first_letter(data):
    return [word.capitalize() for word in data]

# cmumosei四舍五入
def cmumosei_round(a):
    if a < -2:
        res = -3
    if -2 <= a and a < -1:
        res = -2
    if -1 <= a and a < 0:
        res = -1
    if 0 <= a and a <= 0:
        res = 0
    if 0 < a and a <= 1:
        res = 1
    if 1 < a and a <= 2:
        res = 2
    if a > 2:
        res = 3
    return res

# From MTCNN
def fixed_image_standardization(image_tensor: torch.tensor) -> torch.tensor:
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

# 填充tensor
def padTensor(t: torch.tensor, targetLen: int) -> torch.tensor:
    oriLen, dim = t.size()
    return torch.cat((t, torch.zeros(targetLen - oriLen, dim).to(t.device)), dim=0)

# 计算积极所占百分比
def calc_percent(x: torch.tensor):
    total = np.prod(np.array(x.size()))
    positive = x.sum().item()
    return positive / total
