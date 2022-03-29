import copy
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tabulate import tabulate
from src.evaluate import eval_iemocap, eval_iemocap_ce
from src.trainers.basetrainer import TrainerBase
from transformers import AlbertTokenizer

class IemocapTrainer(TrainerBase):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(IemocapTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)
        self.args = args
        self.text_max_len = args['text_max_len']
        self.tokenizer = AlbertTokenizer.from_pretrained(f'albert-{args["text_model_size"]}-v2')
        self.eval_func = eval_iemocap if args['loss'] == 'bce' else eval_iemocap_ce
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []

        # 六种注释的字符串列表
        annotations = dataloaders['train'].dataset.get_annotations()

        if self.args['loss'] == 'bce':
            # 二维列表 就是最后要显示那个表格的数据结构
            self.headers = [
                ['phase (acc)', *annotations, 'average'],
                ['phase (recall)', *annotations, 'average'],
                ['phase (precision)', *annotations, 'average'],
                ['phase (f1)', *annotations, 'average'],
                ['phase (auc)', *annotations, 'average']
            ]

            # n：注释数量+1=6+1=7 1是average
            n = len(annotations) + 1
            # 表格中所有数字都初始化为负无穷
            # 训练集的评估状态 7*5=35
            self.prev_train_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n]
            # 验证集的评估状态 7*5=35
            self.prev_valid_stats = copy.deepcopy(self.prev_train_stats)
            # 测试集的评估状态 7*5=35
            self.prev_test_stats = copy.deepcopy(self.prev_train_stats)
            # 最好的那论验证集的评估状态 7*5=35
            self.best_valid_stats = copy.deepcopy(self.prev_train_stats)

        else:
            self.header = ['Phase', 'Acc', 'Recall', 'Precision', 'F1']
            self.best_valid_stats = [0, 0, 0, 0]

        # 记录最好的那个epoch
        self.best_epoch = -1

    def train(self):
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            train_stats, train_thresholds = self.train_one_epoch()
            valid_stats, valid_thresholds = self.eval_one_epoch()
            # 在一个epoch之后的验证集阈值valid_thresholds 在测试集上做评估 并返回评估的四项结果
            test_stats, _ = self.eval_one_epoch('test', valid_thresholds)
            # test_stats, _ = self.eval_one_epoch('test', [0.5,0.5,0.5,0.5,0.5,0.5])

            print('Train thresholds: ', train_thresholds)
            print('Valid thresholds: ', valid_thresholds)

            # if self.args['model'] == 'mme2e_sparse':
            #     sparse_percentages = self.model.get_sparse_percentages()
            #     if 'v' in self.args['modalities']:
            #         print('V sparse percent', sparse_percentages[0])
            #     if 'a' in self.args['modalities']:
            #         print('A sparse percent', sparse_percentages[1])

            # 将所有的表格都保存起来 每个集都是一个三维的列表了 其中的总数字长度都为 35、35、35
            self.all_train_stats.append(train_stats)
            self.all_valid_stats.append(valid_stats)
            self.all_test_stats.append(test_stats)

            if self.args['loss'] == 'ce':
                train_stats_str = [f'{s:.4f}' for s in train_stats]
                valid_stats_str = [f'{s:.4f}' for s in valid_stats]
                test_stats_str = [f'{s:.4f}' for s in test_stats]
                print(tabulate([
                    ['Train', *train_stats_str],
                    ['Valid', *valid_stats_str],
                    ['Test', *test_stats_str]
                ], headers=self.header))
                if valid_stats[-1] > self.best_valid_stats[-1]:
                    self.best_valid_stats = valid_stats
                    self.best_epoch = epoch
                    self.earlyStop = self.args['early_stop']
                else:
                    self.earlyStop -= 1

            #bce 多标签分类
            else:
                # print('len(self.headers)', len(self.headers)) #5
                for i in range(len(self.headers)): #0-4 self.headers：acc、recall、precision、f1、auc
                    for j in range(len(valid_stats[i])): #valid_stats 验证集的四个评价指标 accs, recalls, precisions, f1s, aucs

                        # is_pivot:True,False
                        # acc:i==0 f1:i==3  auc:i==4
                        # True的条件: 到了average那个位置
                        # False的条件: 所有其他位置
                        # auc average i应该==3 这里原来的代码写错了
                        is_pivot = (i == 0 and j == (len(valid_stats[i]) - 1)) # 3

                        # 当前验证集这个指标的评价数值大于在最佳验证状态那个对应位置的数值
                        if valid_stats[i][j] > self.best_valid_stats[i][j]:
                            # 替换
                            self.best_valid_stats[i][j] = valid_stats[i][j]

                            # is_pivot==true
                            if is_pivot:
                                # 当前发现了最佳的模型状态 参数 那么早停值归为初始值
                                self.earlyStop = self.args['early_stop']
                                # 记录当前epoc为最佳epoch
                                self.best_epoch = epoch
                                # 将最优模型保存
                                self.best_model = copy.deepcopy(self.model.state_dict())

                        # 如果到了那个位置 并且 没有超过最好的记录
                        elif is_pivot:
                            # 早停-1 耐心值-1
                            self.earlyStop -= 1

                    # 向表格中打印数据 并和上一次的结果作比较 打印箭头
                    train_stats_str = self.make_stat(self.prev_train_stats[i], train_stats[i])
                    valid_stats_str = self.make_stat(self.prev_valid_stats[i], valid_stats[i])
                    test_stats_str = self.make_stat(self.prev_test_stats[i], test_stats[i])

                    self.prev_train_stats[i] = train_stats[i]
                    self.prev_valid_stats[i] = valid_stats[i]
                    self.prev_test_stats[i] = test_stats[i]

                    print(tabulate([
                        ['Train', *train_stats_str],
                        ['Valid', *valid_stats_str],
                        ['Test', *test_stats_str]
                    ], headers=self.headers[i]))



            # 早停值为0了结束
            if self.earlyStop == 0:
                break

        print('=== Best performance ===')
        if self.args['loss'] == 'ce':
            print(tabulate([
                [f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1]]
            ], headers=self.header))
        else:
            for i in range(len(self.headers)):
                print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][i]]], headers=self.headers[i]))

        self.save_stats()
        self.save_model()
        print('Results and model are saved!')

    def valid(self):
        valid_stats = self.eval_one_epoch()
        for i in range(len(self.headers)):
            print(tabulate([['Valid', *valid_stats[i]]], headers=self.headers[i]))
            print()

    def test(self):
        test_stats = self.eval_one_epoch('test')
        for i in range(len(self.headers)):
            print(tabulate([['Test', *test_stats[i]]], headers=self.headers[i]))
            print()
        for stat in test_stats:
            for n in stat:
                print(f'{n:.4f},', end='')
        print()

    def train_one_epoch(self):
        self.model.train()
        if self.args['model'] == 'mme2e' or self.args['model'] == 'mme2e_sparse':
            self.model.mtcnn.eval()

        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc='Train')

        # with torch.autograd.set_detect_anomaly(True):
        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce':
                Y = Y.argmax(-1)

            # imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                loss = self.criterion(logits, Y)
                loss.backward()
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)
                if self.args['clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.args['clip'])
                self.optimizer.step()
            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())
            pbar.set_description("train loss:{:.4f}".format(epoch_loss / data_size))
            if self.scheduler is not None:
                self.scheduler.step()
        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)

        epoch_loss /= len(dataloader.dataset)
        # print(f'train loss = {epoch_loss}')
        return self.eval_func(total_logits, total_Y)

    def eval_one_epoch(self, phase='valid', thresholds=None):
        self.model.eval()
        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []
        pbar = tqdm(dataloader, desc=phase)

        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce':
                Y = Y.argmax(-1)

            # imgs = imgs.to(device=self.device)
            specgrams = specgrams.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text) # (batch_size, num_classes)
                loss = self.criterion(logits, Y)
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)

            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())

            pbar.set_description(f"{phase} loss:{epoch_loss/data_size:.4f}")

        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)

        epoch_loss /= len(dataloader.dataset)

        # if phase == 'valid' and self.scheduler is not None:
        #     self.scheduler.step(epoch_loss)

        # print(f'{phase} loss = {epoch_loss}')
        return self.eval_func(total_logits, total_Y, thresholds)
