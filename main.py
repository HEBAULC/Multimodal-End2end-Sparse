import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.cli import get_args
from src.datasets import get_dataset_iemocap, collate_fn, HCFDataLoader, get_dataset_mosei, collate_fn_hcf_mosei, \
    get_dataset_sims
# from src.models.e2e import MME2E
from src.models.e2e_lf_dnn import MME2E_LFDNN
from src.models.sparse_e2e import MME2E_Sparse
from src.models.e2e import MME2E
from src.models.baselines.lf_rnn import LF_RNN
from src.models.baselines.lf_transformer import LF_Transformer
from src.trainers.emotiontrainer import IemocapTrainer
from src.trainers.sentimenttrainer import SimsTrainer

# 主函数 运行的起点
if __name__ == "__main__":
    # 程序开始时间
    start = time.time()

    # 获取超参数
    args = get_args()

    # Fix seed for reproducibility
    # 固定种子以保证重复性
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    # 设置运行的设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    # 使用gpu加速
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 使用cpu进行测试
    # device = torch.device('cpu')
    # device = torch.device(f"cuda:{args['cuda']}" if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(int(args['cuda']))

    print("Start loading the data....")

    # 分支模块1: 处理dataset参数
    if args['dataset'] == 'iemocap':
        train_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='train',
                                            img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        valid_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='valid',
                                            img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])
        test_dataset = get_dataset_iemocap(data_folder=args['datapath'], phase='test',
                                           img_interval=args['img_interval'], hand_crafted_features=args['hand_crafted'])

        if args['hand_crafted']:
            train_loader = HCFDataLoader(dataset=train_dataset, feature_type=args['audio_feature_type'],
                                         batch_size=args['batch_size'], shuffle=True, num_workers=2)
            valid_loader = HCFDataLoader(dataset=valid_dataset, feature_type=args['audio_feature_type'],
                                         batch_size=args['batch_size'], shuffle=False, num_workers=2)
            test_loader = HCFDataLoader(dataset=test_dataset, feature_type=args['audio_feature_type'],
                                        batch_size=args['batch_size'], shuffle=False, num_workers=2)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                      num_workers=2, collate_fn=collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False,
                                      num_workers=2, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,
                                     num_workers=2, collate_fn=collate_fn)

    elif args['dataset'] == 'mosei':
        train_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='train', img_interval=args['img_interval']
                                          , hand_crafted_features=args['hand_crafted'])
        valid_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval']
                                          , hand_crafted_features=args['hand_crafted'])
        test_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='test', img_interval=args['img_interval']
                                         , hand_crafted_features=args['hand_crafted'])

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2
                                  , collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2
                                  , collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2
                                 , collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)

    elif args['dataset'] == 'sims':
        train_dataset = get_dataset_sims(data_folder=args['datapath'], phase='train', img_interval=args['img_interval'],
                                         hand_crafted_features=args['hand_crafted'])
        valid_dataset = get_dataset_sims(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval'],
                                         hand_crafted_features=args['hand_crafted'])
        test_dataset = get_dataset_sims(data_folder=args['datapath'], phase='test', img_interval=args['img_interval'],
                                        hand_crafted_features=args['hand_crafted'])

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2
                                  , collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2
                                  , collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2
                                 , collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)

        # 人工特征
        # train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2,
        #                           collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        # valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2,
        #                           collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)
        # test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2,
        #                          collate_fn=collate_fn_hcf_mosei if args['hand_crafted'] else collate_fn)


    # 打印训练集、验证集、测试集样本数量信息
    print(f'# Train samples = {len(train_loader.dataset)}')
    print(f'# Valid samples = {len(valid_loader.dataset)}')
    print(f'# Test samples = {len(test_loader.dataset)}')

    # dataloaders加载数据
    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    # 通过字典获取learning_rate值
    lr = args['learning_rate']

    # 分支模块2：处理训练模型参数 调用某个模型的超参数
    if args['model'] == 'mme2e':
        model = MME2E(args=args, device=device)
        model = model.to(device=device)

        # When using a pre-trained text modal,
        # you can use text_lr_factor to give a smaller leraning rate to the textual model parts
        # 使用预训练文本模态时，可以使用 text_lr_factor 为文本模型部分提供较小的学习率
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam([
                {'params': model.T.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.t_out.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.V.parameters()},
                {'params': model.v_flatten.parameters()},
                {'params': model.v_transformer.parameters()},
                {'params': model.v_out.parameters()},
                {'params': model.A.parameters()},
                {'params': model.a_flatten.parameters()},
                {'params': model.a_transformer.parameters()},
                {'params': model.a_out.parameters()},
                {'params': model.weighted_fusion.parameters()},
            ], lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'mme2e_dnn':
        model = MME2E_LFDNN(args=args, device=device)
        model = model.to(device=device)

        # When using a pre-trained text modal,
        # you can use text_lr_factor to give a smaller leraning rate to the textual model parts
        # 使用预训练文本模态时，可以使用 text_lr_factor 为文本模型部分提供较小的学习率
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'],
                                         weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam([
                {'params': model.T.parameters(), 'lr': lr / args['text_lr_factor']},
                # {'params': model.t_out.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.V.parameters()},
                # {'params': model.v_flatten.parameters()},
                {'params': model.v_transformer.parameters()},
                # {'params': model.v_out.parameters()},
                {'params': model.A.parameters()},
                # {'params': model.a_flatten.parameters()},
                {'params': model.a_transformer.parameters()},
                # {'params': model.a_out.parameters()},
                {'params': model.weighted_fusion.parameters()},
            ], lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'mme2e_sparse':
        model = MME2E_Sparse(args=args, device=device)
        model = model.to(device=device)

        # When using a pre-trained text modal, you can use text_lr_factor to give a smaller leraning rate to the textual model parts
        # 使用预训练文本模态时，可以使用 text_lr_factor 为文本模型部分提供较小的学习率
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        else:
            optimizer = torch.optim.Adam([
                {'params': model.T.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.t_out.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.V.parameters()},
                {'params': model.v_flatten.parameters()},
                {'params': model.v_transformer.parameters()},
                {'params': model.v_out.parameters()},
                {'params': model.A.parameters()},
                {'params': model.a_flatten.parameters()},
                {'params': model.a_transformer.parameters()},
                {'params': model.a_out.parameters()},
                {'params': model.weighted_fusion.parameters()},
            ], lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'lf_rnn':
        model = LF_RNN(args)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args['weight_decay'])
    elif args['model'] == 'lf_transformer':
        model = LF_Transformer(args)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args['weight_decay'])
    else:
        raise ValueError('Incorrect model name!')

    # 分支模块3：处理scheduler调度器参数 动态学习率
    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'] * len(train_loader.dataset) // args['batch_size'])
    else:
        scheduler = None

    # 分支模块4：处理loss参数 设置损失函数的类型
    if args['loss'] == 'l1':
        criterion = torch.nn.L1Loss()

    elif args['loss'] == 'mse':
        criterion = torch.nn.MSELoss()

    # 单个标签多分类时使用ce
    elif args['loss'] == 'ce':
        criterion = torch.nn.CrossEntropyLoss()

    # 多标签分类时使用bce
    elif args['loss'] == 'bce':
        pos_weight = train_dataset.getPosWeight()
        pos_weight = torch.tensor(pos_weight).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = torch.nn.BCEWithLogitsLoss()

    # 分支模块5：如果是'iemocap' or 'mosei'数据集 调用IemocapTrainer训练函数 训练新数据集时需要自己手动更换
    if args['dataset'] == 'iemocap' or 'mosei' or 'mmsa-mosei':
        trainer = IemocapTrainer(args, model, criterion, optimizer, scheduler, device, dataloaders)
    elif args['dataset'] == 'sims':
        trainer = SimsTrainer(args, model, criterion, optimizer, scheduler, device, dataloaders)

    # if args['dataset'] == 'sims':
    #     # torch.multiprocessing.set_start_method('spawn')
    #     trainer = SimsTrainer(args, model, criterion, optimizer, scheduler, device, dataloaders)

    # 分支模块6:处理训练方式
    if args['test']:
        trainer.test()
    elif args['valid']:
        trainer.valid()
    else:
        trainer.train()

# 程序结束时间
    end = time.time()

# 训练程序运行总耗时间
    print(f'Total time usage = {(end - start) / 3600:.2f} hours.')
