import copy
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tabulate import tabulate
from src.evaluate import eval_iemocap, eval_iemocap_ce, eval_sims_senti
from src.trainers.basetrainer import TrainerBase
from transformers import AlbertTokenizer

class SimsTrainer(TrainerBase):
    # 参数列表、训练的模型、损失函数标准、优化器、衰减学习率、计算设备、数据加载器
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(SimsTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)
        self.args = args
        self.text_max_len = args['text_max_len']
        self.tokenizer = AlbertTokenizer.from_pretrained(f'albert-{args["text_model_size"]}-v2')
        # self.eval_func = eval_iemocap if args['loss'] == 'bce' else eval_iemocap_ce
        self.eval_func = eval_sims_senti
        # 所有的训练集状态
        self.all_train_stats = []
        # 所有的验证集状态
        self.all_valid_stats = []
        # 所有的测试集状态
        self.all_test_stats = []

        annotations = dataloaders['train'].dataset.get_annotations()

        if self.args['loss'] == 'bce':
            self.headers = [
                ['phase (acc)', *annotations, 'average'],
                ['phase (recall)', *annotations, 'average'],
                ['phase (precision)', *annotations, 'average'],
                ['phase (f1)', *annotations, 'average'],
                ['phase (auc)', *annotations, 'average']
            ]

            n = len(annotations) + 1
            self.prev_train_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n]
            self.prev_valid_stats = copy.deepcopy(self.prev_train_stats)
            self.prev_test_stats = copy.deepcopy(self.prev_train_stats)
            self.best_valid_stats = copy.deepcopy(self.prev_train_stats)

        else:
            # self.header = ['Phase', 'Acc', 'Recall', 'Precision', 'F1']
            self.header = ['Phase', 'acc2', 'acc3', 'acc5', 'F1', 'mae', 'corr']
            # 最好的验证集状态
            self.best_valid_stats = [0, 0, 0, 0, 0, 0]

        self.best_epoch = -1

    def train(self):
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            train_stats = self.train_one_epoch() #, train_thresholds
            valid_stats = self.eval_one_epoch() #, valid_thresholds
            test_stats = self.eval_one_epoch('test') #, valid_thresholds, _
            # test_stats, _ = self.eval_one_epoch('test', [0.5,0.5,0.5,0.5,0.5,0.5])

            # print('Train thresholds: ', train_thresholds)
            # print('Valid thresholds: ', valid_thresholds)

            # if self.args['model'] == 'mme2e_sparse':
            #     sparse_percentages = self.model.get_sparse_percentages()
            #     if 'v' in self.args['modalities']:
            #         print('V sparse percent', sparse_percentages[0])
            #     if 'a' in self.args['modalities']:
            #         print('A sparse percent', sparse_percentages[1])

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
                # if valid_stats[-1] > self.best_valid_stats[-1]:
                if valid_stats[4] > self.best_valid_stats[4]:
                    self.best_valid_stats = valid_stats
                    self.best_epoch = epoch
                    self.earlyStop = self.args['early_stop']
                else:
                    self.earlyStop -= 1

            else:
                train_stats_str = [f'{s:.4f}' for s in train_stats]
                valid_stats_str = [f'{s:.4f}' for s in valid_stats]
                test_stats_str = [f'{s:.4f}' for s in test_stats]
                print(tabulate([
                    ['Train', *train_stats_str],
                    ['Valid', *valid_stats_str],
                    ['Test', *test_stats_str]
                ], headers=self.header))
                if valid_stats[3] > self.best_valid_stats[3]:
                    # print('valid_stats', valid_stats)
                    # print('best_valid_stats', self.best_valid_stats)
                    #
                    # print('valid_stats[3]', valid_stats[3])
                    # print('best_valid_stats[3]', self.best_valid_stats[3])
                    self.best_valid_stats = valid_stats
                    self.best_epoch = epoch
                    self.earlyStop = self.args['early_stop']
                else:
                    self.earlyStop -= 1

            # 非ce损失 如bce、mse
            # else:
            #     for i in range(len(self.header)):
            #         for j in range(len(valid_stats[i])):
            #             is_pivot = (i == 3 and j == (len(valid_stats[i]) - 1)) # auc average
            #             if valid_stats[i][j] > self.best_valid_stats[i][j]:
            #                 self.best_valid_stats[i][j] = valid_stats[i][j]
            #                 if is_pivot:
            #                     self.earlyStop = self.args['early_stop']
            #                     self.best_epoch = epoch
            #                     self.best_model = copy.deepcopy(self.model.state_dict())
            #             elif is_pivot:
            #                 self.earlyStop -= 1
            #
            #         train_stats_str = self.make_stat(self.prev_train_stats[i], train_stats[i])
            #         valid_stats_str = self.make_stat(self.prev_valid_stats[i], valid_stats[i])
            #         test_stats_str = self.make_stat(self.prev_test_stats[i], test_stats[i])
            #
            #         self.prev_train_stats[i] = train_stats[i]
            #         self.prev_valid_stats[i] = valid_stats[i]
            #         self.prev_test_stats[i] = test_stats[i]
            #
            #         print(tabulate([
            #             ['Train', *train_stats_str],
            #             ['Valid', *valid_stats_str],
            #             ['Test', *test_stats_str]
            #         ],  headers=self.header[i]))

            if self.earlyStop == 0:
                break

        print('=== Best performance ===')
        if self.args['loss'] == 'ce':
            print(tabulate([
                [f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1]]
            ], headers=self.header))

        else:
            print(tabulate([
                [f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1]]
            ], headers=self.header))
            # for i in range(len(self.headers)):
            #     print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][i]]],
            #                    headers=self.header[i]))

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
        # 所有的logits值
        total_logits = []
        # 所有的标签值
        total_Y = []
        pbar = tqdm(dataloader, desc='Train')

        # with torch.autograd.set_detect_anomaly(True):
        for uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y in pbar:
            # 取出dataloader中的各项
            # print(uttranceId, imgs, imgLens, specgrams, specgramLens, text, Y)

            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            # 如果是交叉熵损失 那么Y取最后一个纬度的最大值下标
            if self.args['loss'] == 'ce':
                # https://blog.csdn.net/weixin_39190382/article/details/105854567
                Y = Y.argmax(-1)

            # imgs = imgs.to(device=self.device)
            # 频谱图转到tensor
            specgrams = specgrams.to(device=self.device)
            # 文本转到tensor
            text = text.to(device=self.device)
            # 标签转tensor
            Y = Y.to(device=self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                #logits的形状 (batch_size, num_classes) torch.Size([8, 1]) batch_size=8 num_classes=1
                logits = self.model(imgs, imgLens, specgrams, specgramLens, text)
                # print("logits.size()=", logits.size())
                # Y是真实标签
                # print("Y.size()=", Y.size())
                # print(Y)
                if self.args['loss'] == 'mse':
                    Y = Y.view(-1, 1)
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
                if self.args['loss'] == 'mse':
                    Y = Y.view(-1, 1)
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