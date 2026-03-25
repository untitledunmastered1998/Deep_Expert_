import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.optim import Adam

from losses.distill_loss import DistillKL
from losses.loss import sup_con_loss
from utils import get_transform
from utils.rotation_transform import RandomFlip

EPSILON = 1e-8


class Dexpert(object):
    def __init__(self, model: nn.Module, buffer, optimizer, input_size, args):
        self.model = model
        self.optimizer = optimizer

        self.ins_t = args.ins_t
        self.epoch = args.epoch
        self.expert = int(args.expert)
        self.n_classes_num = args.n_classes
        self.use_ncm = (args.classifier == 'ncm')

        self.buffer = buffer
        self.buffer_per_class = 7
        self.buffer_batch_size = args.buffer_batch_size
        self.buffer_cur_task = (self.buffer_batch_size // 2) - args.batch_size
        self.nums_expert = args.nums_expert

        if args.dataset == "cifar10":
            self.total_samples = 10000
        elif "cifar100" in args.dataset:
            self.total_samples = 5000
        elif args.dataset == "tiny_imagenet":
            self.total_samples = 1000
        elif args.dataset == "imagenet_1k":
            self.total_samples = 10000
        self.print_num = self.total_samples // 10

        transform_names = ['simclr', 'ocm', 'scr', 'policy']
        # transform_names = ['simclr', 'policy']
        # if args.nums_expert == 6:
        #     transform_names = ['simclr', 'ocm', 'scr', 'policy', 'policy1', 'policy2']
        # elif args.nums_expert == 8:
        #     transform_names = ['simclr', 'ocm', 'scr', 'policy', 'policy1', 'policy2', 'policy3', 'policy4']
        self.transforms_lists = [get_transform(name, input_size) for name in transform_names]

        self.total_step = 0
        self.class_holder = []
        self.old_class_holder = []
        self.scaler = GradScaler()
        self.args = args
        self.class_per_task = args.class_per_task
        self.known_classes = 0
        self.total_classes = 0
        self.reinit_optimizer = args.reinit_optimizer
        self.temp = args.temp
        self.old_model = None
        self.kd_func = DistillKL(self.temp)
        self.dist_weight = 2 * self.total_samples / self.args.buffer_size

    def train_any_task(self, task_id, train_loader, epoch, feat_id=None):
        num_d = 0
        epoch_log_holder = []
        if epoch == 0:
            self.new_class_holder = []

        for batch_idx, (x, y) in enumerate(train_loader):
            num_d += x.shape[0]

            Y = deepcopy(y)
            for j in range(len(Y)):
                if Y[j] not in self.class_holder:
                    self.class_holder.append(Y[j].detach().item())
                    self.new_class_holder.append(Y[j].detach().item())

            loss = 0.
            mse_loss = 0.
            cosine_loss = 0.
            div_loss = 0.
            dis_loss = 0.
            sdl_loss = 0.
            loss_log = {
                'step': self.total_step,
                'train/loss': 0.,
                'train/ins': 0.,
                'train/ce': 0.,
                'train/dis': 0.,
                'train/sdl': 0.,
            }

            if len(self.buffer) > 0:

                with autocast():
                    # x: [batch_size, 3, input_size, input_size], dtype: torch.Tensor
                    # y: [batch_size], dtype: torch.Tensor
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

                    # sample enough new class samples
                    new_x = x.detach()
                    new_y = y.detach()
                    if batch_idx != 0:
                        # self.buffer_batch_size: 64, self.buffer_cur_task: 22
                        buffer_cur_task = self.buffer_batch_size if task_id == 0 else self.buffer_cur_task
                        cur_x, cur_y, _ = self.buffer.onlysample(buffer_cur_task, task=task_id)
                        if len(cur_x.shape) > 3:
                            new_x = torch.cat((x.detach(), cur_x))
                            new_y = torch.cat((y.detach(), cur_y))

                    if task_id > 0:
                        # balanced sampling for an ideal overall distribution
                        new_over_all = len(self.new_class_holder) / len(self.class_holder)
                        new_batch_size = min(int(self.buffer_batch_size * new_over_all), x.size(0))
                        buffer_batch_size = min(
                            self.buffer_batch_size - new_batch_size,
                            self.buffer_per_class * len(self.class_holder)
                        )
                        mem_x, mem_y, bt = self.buffer.sample(buffer_batch_size, exclude_task=task_id)
                        cat_x = torch.cat((x[:new_batch_size].detach(), mem_x))
                        cat_y = torch.cat((y[:new_batch_size].detach(), mem_y))

                        # cat_x: mixed current task samples and buffer samples

                        # rotate and augment
                        new_x = RandomFlip(new_x, 2)
                        new_y = new_y.repeat(2)
                        cat_x = RandomFlip(cat_x, 2)
                        cat_y = cat_y.repeat(2)

                        new_y = torch.cat((new_y, new_y))
                        cat_y = torch.cat((cat_y, cat_y))

                        new_input_size = new_y.size(0)
                        cat_input_size = cat_y.size(0)
                        mem_input_size = cat_input_size - new_batch_size

                        all_y = torch.cat((new_y, cat_y))
                        new_y = new_y.detach()
                        cat_y = cat_y.detach()
                        all_y = all_y.detach()

                        proj_feat_list = []
                        pred_list = []
                        old_pred_list = []
                        expert_feat_list = []

                        for i in range(self.nums_expert):
                            new_x_i = torch.cat((new_x, self.transforms_lists[i](new_x)))
                            cat_x_i = torch.cat((cat_x, self.transforms_lists[i](cat_x)))

                            all_x = torch.cat((new_x_i, cat_x_i))
                            all_x = all_x.detach()

                            general_feat, expert_feat = self.model.features(all_x, idx=i)
                            proj_feat = self.model.forward_expert_proj_head(expert_feat, idx=i)
                            pred = self.model.forward_expert_head(expert_feat, idx=i)

                            with torch.no_grad():
                                old_general_feat, old_expert_feat = self.old_model.features(all_x, idx=i)
                                old_pred = self.old_model.forward_expert_head(old_expert_feat, idx=i)

                            proj_feat_list.append(proj_feat)
                            pred_list.append(pred)
                            old_pred_list.append(old_pred)
                            expert_feat_list.append(expert_feat)

                        teacher_feat = expert_feat_list[feat_id]
                        teacher_feat = teacher_feat.detach()

                        for i in range(self.nums_expert):
                            ins_loss = sup_con_loss(proj_feat_list[i], self.ins_t, all_y)

                            cat_pred = pred_list[i][new_input_size:]
                            new_pred = pred_list[i][:new_input_size]

                            ce_loss = 2 * F.cross_entropy(cat_pred[:, :self.total_classes], cat_y)

                            fake_targets = new_y - self.known_classes
                            ce_loss += F.cross_entropy(new_pred[:, self.known_classes:], fake_targets)

                            dis_loss = self.dist_weight * self.kd_func(pred_list[i][:, :self.known_classes],
                                                                       old_pred_list[i])

                            sdl_loss = 0.
                            if i != feat_id:
                                sdl_loss = self.args.sdl_weight * torch.dist(
                                    F.normalize(expert_feat_list[i][:new_input_size], dim=1),
                                    F.normalize(teacher_feat[:new_input_size], dim=1), p=2
                                )

                            loss += ins_loss + ce_loss + dis_loss + sdl_loss

                            loss_log['train/ins'] += ins_loss.item() if ins_loss != 0. else 0.
                            loss_log['train/ce'] += ce_loss.item() if ce_loss != 0. else 0.
                            loss_log['train/dis'] += dis_loss.item() if dis_loss != 0. else 0.
                            loss_log['train/sdl'] += sdl_loss.item() if sdl_loss != 0. else 0.
                    else:
                        # rotate and augment
                        new_x = RandomFlip(new_x, 2)
                        new_y = new_y.repeat(2)

                        new_y = torch.cat((new_y, new_y))
                        new_y = new_y.detach()

                        proj_feat_list = []
                        pred_list = []

                        for i in range(self.nums_expert):
                            new_x_i = torch.cat((new_x, self.transforms_lists[i](new_x)))
                            new_x_i = new_x_i.detach()

                            general_feat, expert_feat = self.model.forward_expert_features(new_x_i, idx=i)
                            proj_feat = self.model.forward_expert_proj_head(expert_feat, idx=i)
                            pred = self.model.forward_expert_head(expert_feat, idx=i)

                            proj_feat_list.append(proj_feat)
                            pred_list.append(pred)

                        for i in range(self.nums_expert):
                            ins_loss = sup_con_loss(proj_feat_list[i], self.ins_t, new_y)
                            ce_loss = F.cross_entropy(pred_list[i], new_y)

                            loss += ins_loss + ce_loss

                            loss_log['train/ins'] += ins_loss.item() if ins_loss != 0. else 0.
                            loss_log['train/ce'] += ce_loss.item() if ce_loss != 0. else 0.
                            loss_log['train/dis'] += dis_loss.item() if dis_loss != 0. else 0.
                            loss_log['train/sdl'] += sdl_loss.item() if sdl_loss != 0. else 0.

                if batch_idx % 10 == 0:
                    print("loss", loss)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if epoch == 0:
                self.buffer.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=task_id)

            loss_log['train/loss'] = loss.item() if loss != 0. else 0.
            epoch_log_holder.append(loss_log)
            self.total_step += 1

            if num_d % self.print_num == 0 or batch_idx == 1:
                print(f"==>>> it: {batch_idx}, loss: ins {loss_log['train/ins']:.2f} + ce {loss_log['train/ce']:.3f} + dis {loss_log['train/dis']:.3f} + dis {loss_log['train/sdl']:.3f}, {100 * (num_d / self.total_samples)}%")

        return epoch_log_holder

    def train(self, task_id, train_loader):
        if task_id > 0:
            leep_scores = []
            for feat in range(self.nums_expert):
                leep_score = self.leep(
                    feat_id=feat,
                    data_loader=train_loader,
                    number_of_target_labels=self.class_per_task
                )
                leep_scores.append(leep_score)
                print(f"Expert {feat} LEEP score: {leep_score:.4f}")
            feat_id = np.argmax(leep_scores)
            print(f"Selected expert {feat_id} with LEEP score {leep_scores[feat_id]:.4f}")

            self.known_class_holder = deepcopy(self.class_holder)
        else:
            feat_id = None
        # feat_id = None
        self.pre_train(task_id)
        self.model.train()
        train_log_holder = []
        for epoch in range(self.epoch):
            epoch_log_holder = self.train_any_task(task_id, train_loader, epoch, feat_id)
            train_log_holder.extend(epoch_log_holder)
            # self.buffer.print_per_task_num()
        self.after_train(task_id)
        return train_log_holder

    def pre_train(self, task_id):
        self.total_classes += self.class_per_task
        self.model.add_head(self.class_per_task, bias=self.args.fc_bias)
        if self.reinit_optimizer == 'on':
            self.optimizer = Adam(self.model.parameters(), self.args.lr, weight_decay=self.args.wd)

    def distill_loss(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def after_train(self, task_id):
        self.known_classes += self.class_per_task
        if self.args.use_dummy_cls == 'on' and self.args.dummy_cls > 0 and task_id > 0:
            for i in range(self.nums_expert):
                del self.model.multi_heads[i][-1]
        torch.save(self.model.state_dict(), os.path.join(self.args.log_path, f"model_{task_id}.pth"))
        if self.args.compensate == 'on' or self.args.dist_weight > 0:
            self.old_model = deepcopy(self.model)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False

    def test(self, i, task_loader):
        self.model.eval()
        if self.use_ncm:
            # calculate the class means for each feature layer
            print("\nCalculate class means for last layer...\n")
            print(self.buffer.bx.shape)
            self.class_means_ls = [{} for _ in range(self.nums_expert)]
            class_inputs = {cls: [] for cls in self.class_holder}
            for x, y in zip(self.buffer.x, self.buffer.y_int):
                class_inputs[y.item()].append(x)

            for y in self.class_holder:
                if y not in self.new_class_holder:
                    self.old_class_holder.append(y)

            extracted_current_features = [[] for _ in range(self.nums_expert)]
            for cls, inputs in class_inputs.items():
                features = [[] for _ in range(self.nums_expert)]
                for ex in inputs:
                    return_features_ls = self.model.features(ex.unsqueeze(0))
                    for feat_id in range(self.nums_expert):
                        feature = return_features_ls[feat_id].detach().clone()
                        feature = F.normalize(feature, dim=1)
                        features[feat_id].append(feature.squeeze())
                        extracted_current_features[feat_id].append(feature.squeeze())

                for feat_id in range(self.nums_expert):
                    if len(features[feat_id]) == 0:
                        mu_y = torch.normal(0, 1,
                                            size=tuple(self.model.features(x.unsqueeze(0))[feat_id].detach().size()))
                        mu_y = mu_y.to(x.device)
                    else:
                        features[feat_id] = torch.stack(features[feat_id])
                        mu_y = features[feat_id].mean(0)

                    mu_y = F.normalize(mu_y.reshape(1, -1), dim=1)
                    self.class_means_ls[feat_id][cls] = mu_y.squeeze()

            extracted_current_features = [torch.stack(extracted_current_features[feat_id]).cpu() for feat_id in
                                          range(self.nums_expert)]

            if i > 0 and self.args.compensate == 'on':
                extracted_old_features = [[] for _ in range(self.nums_expert)]
                with torch.no_grad():
                    for cls, inputs in class_inputs.items():
                        for x in inputs:
                            return_old_features_ls = self.old_model.features(x.unsqueeze(0))
                            for feat_id in range(self.nums_expert):
                                old_features = return_old_features_ls[feat_id].detach().clone()
                                old_features = F.normalize(old_features, dim=1)
                                extracted_old_features[feat_id].append(old_features.squeeze())

                extracted_old_features = [torch.stack(extracted_old_features[feat_id]).cpu() for feat_id in
                                          range(self.nums_expert)]
                # DY = extracted_current_features - extracted_old_features
                DY = [extracted_current_features[feat_id] - extracted_old_features[feat_id] for feat_id in
                      range(self.nums_expert)]
                means = []
                for e in range(self.nums_expert):
                    mean = []
                    for cls, inputs in class_inputs.items():
                        mean.append(self.class_means_ls[e][cls])
                    mean = torch.stack(mean).cpu()
                    means.append(mean)
                # means = [torch.stack(self.class_means_ls[feat_id]).cpu() for feat_id in range(self.nums_expert)]
                num_old_cls = self.known_classes
                means_old = [means[feat_id][:num_old_cls] for feat_id in range(self.nums_expert)]
                for feat_id in range(self.nums_expert):
                    distance = np.sum((np.tile(extracted_old_features[feat_id][None, :, :],
                                               [means_old[feat_id].shape[0], 1, 1]) - np.tile(
                        means_old[feat_id][:, None, :], [1, extracted_old_features[feat_id].shape[0], 1])) ** 2, axis=2)
                    W = np.exp(-distance / (2 * self.args.sigma ** 2)) + 1e-5
                    W_norm = W / np.tile(W.sum(axis=1)[:, None], [1, W.shape[1]])
                    displacement = np.sum(
                        np.tile(W_norm[:, :, None], [1, 1, DY[feat_id].shape[1]]) * np.tile(DY[feat_id][None, :, :],
                                                                                            [W.shape[0], 1, 1]), axis=1)
                    means_update = means_old[feat_id] + displacement
                    for old_cls in self.old_class_holder:
                        idx = self.class_holder.index(old_cls)
                        self.class_means_ls[feat_id][old_cls] = means_update[idx].cuda()

        all_acc_list = {'step': self.total_step}

        # test classifier from each expert
        # for feat_id in range(self.nums_expert):
        #     print(f"{'*'*100}\nTest with the output of expert {feat_id+1}:\n")
        #     with torch.no_grad():
        #         acc_list = np.zeros(len(task_loader))
        #         for j in range(i + 1):
        #             acc = self.test_model(task_loader[j]['test'], j, feat_id=feat_id)
        #             acc_list[j] = acc.item()
        #
        #         all_acc_list[str(feat_id)] = acc_list
        #         # print(f"tasks acc:{acc_list}")
        #         print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        acc_list = np.zeros(len(task_loader))
        all_acc_list[str(3)] = acc_list

        # test mean classifier
        print(f"{'*' * 100}\nTest with the mean dists output of each expert:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model_mean(task_loader[j]['test'], j)
                acc_list[j] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i + 1].mean()}")

        return acc_list, all_acc_list

    def test_model_linear(self, loader, i):
        # test linear classifier
        self.model.eval()
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()

            features = self.model.features(x)
            pred = self.model.head(features, use_proj=False)
            pred = pred.data.max(1, keepdim=True)[1]

            num += x.size()[0]
            correct += pred.eq(y.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_buffer(self, i, task_loader, feat_ids=[0, 1, 2, 3]):
        self.model.eval()
        all_acc_list = {'step': self.total_step}
        # test classifier from each required layer
        for feat_id in feat_ids:
            print(f"{'*' * 100}\nTest with the output of layer: {feat_id + 1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                for j in range(i + 1):
                    acc = self.test_buffer_task(j, feat_id=feat_id)
                    acc_list[j] = acc.item()

                all_acc_list[str(feat_id)] = acc_list
                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i + 1].mean()}")

        # test mean classifier
        print(f"{'*' * 100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_buffer_task_mean(j)
                acc_list[j] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i + 1].mean()}")

        return acc_list, all_acc_list

    def test_buffer_task(self, i, feat_id):
        # test specific layer's output
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()

        x_i, y_i, _ = self.buffer.onlysample(self.buffer.current_index, task=i)

        if self.use_ncm:
            class_means = self.class_means_ls[feat_id]
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                features = self.model.features(x)[feat_id]
                features = F.normalize(features, dim=1)
                features = features.unsqueeze(2)
                means = torch.stack([class_means[cls] for cls in self.class_holder])
                means = torch.stack([means] * x.size(0))
                means = means.transpose(1, 2)
                features = features.expand_as(means)
                dists = (features - means).pow(2).sum(1).squeeze(1)
                pred = dists.min(1)[1]
                pred = torch.Tensor(self.class_holder)[pred].to(x.device)

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()
        else:
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                pred = self.model(x)[feat_id]
                pred = pred.data.max(1, keepdim=True)[1]

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Buffer test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_buffer_task_mean(self, i):
        # test with mean dists for all layers
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()

        x_i, y_i, _ = self.buffer.onlysample(self.buffer.current_index, task=i)

        if self.use_ncm:
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                features_ls = self.model.features(x)
                dists_ls = []

                for feat_id in range(4):
                    class_means = self.class_means_ls[feat_id]
                    features = features_ls[feat_id]
                    features = F.normalize(features, dim=1)
                    features = features.unsqueeze(2)
                    means = torch.stack([class_means[cls] for cls in self.class_holder])
                    means = torch.stack([means] * x.size(0))
                    means = means.transpose(1, 2)
                    features = features.expand_as(means)
                    dists = (features - means).pow(2).sum(1).squeeze(1)
                    dists_ls.append(dists)

                dists_ls = torch.cat([dists.unsqueeze(1) for dists in dists_ls], dim=1)
                dists = dists_ls.mean(dim=1).squeeze(1)
                pred = dists.min(1)[1]
                pred = torch.Tensor(self.class_holder)[pred].to(x.device)

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()

        else:
            for x, y in zip(x_i, y_i):
                x = x.unsqueeze(0).detach()
                y = y.unsqueeze(0).detach()

                pred = self.model(x)
                pred = torch.stack(pred, dim=1)
                pred = pred.mean(dim=1).squeeze(1)
                pred = pred.data.max(1, keepdim=True)[1]

                num += x.size()[0]
                correct += pred.eq(y.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Buffer test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_train(self, i, task_loader, feat_ids=[0, 1, 2, 3]):
        # train accuracy of current task i
        self.model.eval()
        all_acc_list = {'step': self.total_step}

        # test classifier from each required layer
        for feat_id in feat_ids:
            print(f"{'*' * 100}\nTest with the output of layer: {feat_id + 1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                acc = self.test_model(task_loader[i]['train'], i, feat_id=feat_id)
                acc_list[i] = acc.item()

                all_acc_list[str(feat_id)] = acc_list
                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i + 1].mean()}")

        # test mean classifier
        print(f"{'*' * 100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            acc = self.test_model_mean(task_loader[i]['train'], i)
            acc_list[i] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i + 1].mean()}")

        return acc_list, all_acc_list

    def test_model(self, loader, i, feat_id=None):
        # test specific layer's output
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        if self.use_ncm:
            class_means = self.class_means_ls[feat_id]
            class_holder_tensor = torch.tensor(self.class_holder).cuda()
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()

                _, features = self.model.features(data, idx=feat_id)
                features = F.normalize(features, dim=1)
                features = features.unsqueeze(2)
                means = torch.stack([class_means[cls] for cls in self.class_holder])
                # print("stack means shape:", means.shape)
                means = torch.stack([means] * data.size(0))
                # print("stack data shape:", means.shape)
                means = means.transpose(1, 2)
                # print("transpose shape:", means.shape)
                features = features.expand_as(means)
                dists = (features - means).pow(2).sum(1).squeeze()
                pred = dists.min(1)[1]
                # pred = torch.Tensor(self.class_holder)[pred].to(data.device)

                pred = class_holder_tensor[pred]

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        else:
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()

                pred = self.model(data)
                pred = pred.data.max(1, keepdim=True)[1]

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_model_mean(self, loader, i, feat_id=None):
        # test with mean dists for all layers
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        if self.use_ncm:
            class_holder_tensor = torch.tensor(self.class_holder).cuda()
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()
                features_ls = self.model.features(data)
                dists_ls = []

                for feat_id in range(self.nums_expert):
                    class_means = self.class_means_ls[feat_id]
                    features = features_ls[feat_id]
                    features = F.normalize(features, dim=1)
                    features = features.unsqueeze(2)
                    means = torch.stack([class_means[cls] for cls in self.class_holder])
                    means = torch.stack([means] * data.size(0))
                    means = means.transpose(1, 2)
                    features = features.expand_as(means)
                    dists = (features - means).pow(2).sum(1).squeeze()
                    dists_ls.append(dists)

                dists_ls = torch.cat([dists.unsqueeze(1) for dists in dists_ls], dim=1)
                dists = dists_ls.mean(dim=1).squeeze(1)
                pred = dists.min(1)[1]
                # pred = torch.Tensor(self.class_holder)[pred].to(data.device)
                pred = class_holder_tensor[pred]

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        else:
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()

                pred = self.model(data)
                pred = torch.stack(pred, dim=1)
                pred = pred.mean(dim=1).squeeze()
                pred = pred.data.max(1, keepdim=True)[1]

                num += data.size()[0]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def save_checkpoint(self, save_path='./outputs/final.pt'):
        print(f"Save checkpoint to: {save_path}")
        ckpt_dict = {
            'model': self.model.state_dict(),
            'buffer': self.buffer.state_dict(),
        }
        folder, file_name = os.path.split(save_path)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save(ckpt_dict, save_path)

    def load_checkpoint(self, load_path='./outputs/final.pt'):
        print(f"Load checkpoint from: {load_path}")
        ckpt_dict = torch.load(load_path)
        self.model.load_state_dict(ckpt_dict['model'])
        self.buffer.load_state_dict(ckpt_dict['buffer'])

    def leep(self,
             feat_id: int,
             data_loader: torch.utils.data.DataLoader,
             number_of_target_labels: int) -> float:
        assert data_loader.drop_last is False
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            predicted_dataset_length = len(data_loader) * data_loader.batch_size

            original_output_shape = self.known_classes

            print("number of upstream dataset classes: ", original_output_shape)

            # Allocate empty arrays ahead of time

            # Omega from Eq(1) and Eq(2)
            categorical_probability = torch.zeros((predicted_dataset_length, original_output_shape),
                                                  dtype=torch.float32, device=device)

            all_labels = torch.zeros(predicted_dataset_length, dtype=torch.int64, device=device)

            # Joint porbability from Eq (1)
            p_target_label_and_source_distribution = torch.zeros(number_of_target_labels, original_output_shape,
                                                                 device=device)

            soft_max = torch.nn.LogSoftmax()

            # This calculates actual dataset length
            actual_dataset_length = 0

            for i, (images, labels) in enumerate(data_loader):
                current_batch_length = labels.shape[0]
                actual_dataset_length += current_batch_length

                labels -= self.known_classes

                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                general_feat, expert_feat = self.model.forward_expert_features(images, idx=feat_id)
                result = self.model.forward_expert_head(expert_feat, idx=feat_id)
                result = torch.exp(soft_max(result))

                categorical_probability[
                i * data_loader.batch_size:i * data_loader.batch_size + current_batch_length] = result
                all_labels[i * data_loader.batch_size:i * data_loader.batch_size + current_batch_length] = labels
                p_target_label_and_source_distribution[labels] += result.squeeze()

            # Shrink tensors to actually fit to the actual dataset length
            categorical_probability = torch.narrow(categorical_probability, dim=0, start=0,
                                                   length=actual_dataset_length)
            all_labels = torch.narrow(all_labels, dim=0, start=0, length=actual_dataset_length)

            p_target_label_and_source_distribution /= actual_dataset_length
            p_marginal_z_distribution = torch.sum(p_target_label_and_source_distribution, axis=0)
            p_empirical_conditional_distribution = torch.div(p_target_label_and_source_distribution,
                                                             p_marginal_z_distribution)

            total_sum = torch.sum(torch.log(
                torch.sum((p_empirical_conditional_distribution[all_labels] * categorical_probability), axis=1)))
            return (total_sum / actual_dataset_length).item()
