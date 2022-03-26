import argparse

def get_args():
    # 描述：用于情感识别的多模式端到端稀疏模型
    parser = argparse.ArgumentParser(description='Multimodal End-to-End Sparse Model for Emotion Recognition')
    # 超参数最多有38个

    # Training hyper-parameters
    # 训练的超参数 12个
    # 批量大小
    parser.add_argument('-bs', '--batch-size', help='Batch size', type=int, required=True)
    # 学习率
    parser.add_argument('-lr', '--learning-rate', help='Learning rate', type=float, required=True)
    # 权重衰减
    parser.add_argument('-wd', '--weight-decay', help='Weight decay', type=float, required=False, default=0.0)
    # 训练轮数
    parser.add_argument('-ep', '--epochs', help='Number of epochs', type=int, required=True)
    # 早停机制 防止过拟合
    parser.add_argument('-es', '--early-stop', help='Early stop', type=int, required=False, default=5)
    # cuda设备号
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='0')
    # 梯度裁剪 防止梯度爆炸
    parser.add_argument('-cl', '--clip', help='Use clip to gradients', type=float, required=False, default=-1.0)
    # 调度器-衰减学习率
    parser.add_argument('-sc', '--scheduler', help='Use scheduler to optimizer', action='store_true')
    # 随机数种子
    parser.add_argument('-se', '--seed', help='Random seed', type=int, required=False, default=0)
    # 损失函数
    parser.add_argument('--loss', help='loss function', type=str, required=False, default='bce')#default='bce'
    # 优化方法adam/sgd
    parser.add_argument('--optim', help='optimizer function: adam/sgd', type=str, required=False, default='adam')
    # 考虑文本模型的学习率
    parser.add_argument('--text-lr-factor', help='Factor the learning rate of text model', type=int, required=False, default=10)

    # Model
    # 模型的超参数 10个
    # 选择的模型 默认mme2e
    parser.add_argument('-mo', '--model', help='Which model', type=str, required=False, default='mme2e')
    # 预训练文本模型的大小 默认base
    parser.add_argument('--text-model-size', help='Size of the pre-trained text model', type=str, required=False, default='base')
    # 如何融合模态 原来默认早期融合
    # 明明代码实现的是后期融合 但是这里默认原来写的early 可能是作者写错了
    parser.add_argument('--fusion', help='How to fuse modalities', type=str, required=False, default='late')
    # 每个模态模型输出的特征维度 默认256
    parser.add_argument('--feature-dim', help='Dimension of features outputed by each modality model', type=int, required=False, default=256)
    # 稀疏 CNN 层的阈值 default=0.9
    parser.add_argument('-st', '--sparse-threshold', help='Threshold of sparse CNN layers', type=float, required=False, default=0.9)
    # 手工制作的特征尺寸 default=[300, 144, 35]
    parser.add_argument('-hfcs', '--hfc-sizes', help='Hand crafted feature sizes', nargs='+', type=int, required=False, default=[300, 144, 35])
    # CNN后transformer的大小 default=512
    parser.add_argument('--trans-dim', help='Dimension of the transformer after CNN', type=int, required=False, default=512)
    # CNN后transformer的层数  default=2
    parser.add_argument('--trans-nlayers', help='Number of layers of the transformer after CNN', type=int, required=False, default=2)
    # CNN后transformer的头数  default=8
    parser.add_argument('--trans-nheads', help='Number of heads of the transformer after CNN', type=int, required=False, default=8)
    # 手工制作的音频特征类型  default=0
    parser.add_argument('-aft', '--audio-feature-type', help='Hand crafted audio feature types', type=int, default=0)

    # Data
    # 数据的超参数 4个
    # 数据中的情绪数量 分类数 默认是4
    parser.add_argument('--num-emotions', help='Number of emotions in data', type=int, required=False, default=4)
    # 采样图像帧的间隔 default=500
    parser.add_argument('--img-interval', help='Interval to sample image frames', type=int, required=False, default=500)
    # 使用手工制作的功能
    parser.add_argument('--hand-crafted', help='Use hand crafted features', action='store_true')
    # 标记化后的最大文本长度 default=300
    parser.add_argument('--text-max-len', help='Max length of text after tokenization', type=int, required=False, default=300)

    # Path
    # 路径超参数 2个
    # 数据移到文件夹外后使用绝对路经 默认使用的数据集是iemocap 使用其他数据集的时候自己备注
    # 数据路经
    parser.add_argument('--datapath', help='Path of data', type=str, required=False, default='/home/luocong/projects/data/data')
    # 数据集
    parser.add_argument('--dataset', help='Use which dataset', type=str, required=False, default='iemocap')# #sims

    # Evaluation超参数 3个
    # 评估 验证集 测试集
    # 使用哪几种模态 默认tav三种模态
    parser.add_argument('-mod', '--modalities', help='what modalities to use', type=str, required=False, default='tav')
    parser.add_argument('--valid', help='Only run validation', action='store_true')
    parser.add_argument('--test', help='Only run test', action='store_true')

    # Checkpoint
    # 检查点超参数 2个
    # 检查点路径
    parser.add_argument('--ckpt', help='Path of checkpoint', type=str, required=False, default='')
    # 加载检查点的哪种模式
    parser.add_argument('--ckpt-mod', help='Load which modality of the checkpoint', type=str, required=False, default='tav')

    # LSTM超参数 5个
    parser.add_argument('-dr', '--dropout', help='dropout', type=float, required=False, default=0.1)
    parser.add_argument('-nl', '--num-layers', help='num of layers of LSTM', type=int, required=False, default=1)
    parser.add_argument('-hs', '--hidden-size', help='hidden vector size of LSTM', type=int, required=False, default=300)
    parser.add_argument('-bi', '--bidirectional', help='Use Bi-LSTM', action='store_true')
    parser.add_argument('--gru', help='Use GRU rather than LSTM', action='store_true')

    args = vars(parser.parse_args())
    return args
