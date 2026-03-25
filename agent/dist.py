# import os
# from copy import deepcopy

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch.cuda.amp import GradScaler
# from torch.cuda.amp import autocast as autocast

# from losses.loss import sup_con_loss
# from utils import get_transform
# from utils.rotation_transform import RandomFlip


# class Dist(object):
#     def __init__(self, model:nn.Module, buffer, optimizer, input_size, args):
#         self.model = model
#         self.optimizer = optimizer

#         self.ins_t = args.ins_t
#         self.epoch = args.epoch
#         self.expert = int(args.expert)
#         self.n_classes_num = args.n_classes
#         self.use_ncm = (args.classifier == 'ncm')

#         self.buffer = buffer
#         self.buffer_per_class = 7
#         self.buffer_batch_size = args.buffer_batch_size
#         self.buffer_cur_task = (self.buffer_batch_size // 2) - args.batch_size

#         if args.dataset == "cifar10":
#             self.total_samples = 10000
#         elif "cifar100" in args.dataset:
#             self.total_samples = 5000
#         elif args.dataset == "tiny_imagenet":
#             self.total_samples = 1000
#         self.print_num = self.total_samples // 10

#         self.transform = get_transform(args.augmentation, input_size)

#         self.total_step = 0
#         self.class_holder = []
#         self.scaler = GradScaler()
#         self.args = args
#         self.class_per_task = args.class_per_task
#         self.known_classes = 0
#         self.total_classes = 0
#         self.reinit_optimizer = args.reinit_optimizer

#     def train_any_task(self, task_id, train_loader, epoch):
#         num_d = 0
#         epoch_log_holder = []
#         if epoch == 0:
#             self.new_class_holder = []

#         for batch_idx, (x, y) in enumerate(train_loader):
#             num_d += x.shape[0]

#             Y = deepcopy(y)
#             for j in range(len(Y)):
#                 if Y[j] not in self.class_holder:
#                     self.class_holder.append(Y[j].detach().item())
#                     self.new_class_holder.append(Y[j].detach().item())

#             loss = 0.
#             loss_log = {
#                 'step':     self.total_step,
#                 'train/loss':     0.,
#                 'train/ins':      0.,
#                 'train/ce':       0.,
#             }

#             if len(self.buffer) > 0:

#                 with autocast():
#                     # x: [batch_size, 3, input_size, input_size], dtype: torch.Tensor
#                     # y: [batch_size], dtype: torch.Tensor
#                     x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

#                     # sample enough new class samples
#                     new_x = x.detach()
#                     new_y = y.detach()
#                     if batch_idx != 0:
#                         # self.buffer_batch_size: 64, self.buffer_cur_task: 22
#                         buffer_cur_task = self.buffer_batch_size if task_id==0 else self.buffer_cur_task
#                         cur_x, cur_y, _ = self.buffer.onlysample(buffer_cur_task, task=task_id)
#                         if len(cur_x.shape) > 3:
#                             new_x = torch.cat((x.detach(), cur_x))
#                             new_y = torch.cat((y.detach(), cur_y))

#                     if task_id > 0:
#                         # balanced sampling for an ideal overall distribution
#                         new_over_all = len(self.new_class_holder) / len(self.class_holder)
#                         new_batch_size = min(
#                             int(self.buffer_batch_size * new_over_all), x.size(0)
#                         )
#                         buffer_batch_size = min(
#                             self.buffer_batch_size - new_batch_size,
#                             self.buffer_per_class * len(self.class_holder)
#                         )
#                         mem_x, mem_y, bt = self.buffer.sample(buffer_batch_size, exclude_task=task_id)
#                         cat_x = torch.cat((x[:new_batch_size].detach(), mem_x))
#                         cat_y = torch.cat((y[:new_batch_size].detach(), mem_y))

#                         # rotate and augment
#                         new_x = RandomFlip(new_x, 2)
#                         new_y = new_y.repeat(2)
#                         cat_x = RandomFlip(cat_x, 2)
#                         cat_y = cat_y.repeat(2)

#                         new_x = torch.cat((new_x, self.transform(new_x)))
#                         new_y = torch.cat((new_y, new_y))
#                         cat_x = torch.cat((cat_x, self.transform(cat_x)))
#                         cat_y = torch.cat((cat_y, cat_y))

#                         new_input_size = new_x.size(0)
#                         cat_input_size = cat_x.size(0)

#                         all_x = torch.cat((new_x, cat_x))
#                         all_y = torch.cat((new_y, cat_y))
#                         all_x = all_x.detach()
#                         all_y = all_y.detach()

#                         features = self.model.features(all_x)
#                         proj_feat = self.model.head(features, use_proj=True)
#                         pred_out = self.model.head(features, use_proj=False)

#                         ins_loss = sup_con_loss(proj_feat, self.ins_t, all_y)

#                         new_pred = pred_out[:new_input_size]
#                         cat_pred = pred_out[new_input_size:]

#                         ce_loss = 2 * F.cross_entropy(cat_pred, cat_y)

#                         # new_pred = new_pred[:, self.new_class_holder]
#                         # new_y_onehot = F.one_hot(new_y, self.n_classes_num)
#                         # new_y_onehot = new_y_onehot[:, self.new_class_holder].float()
#                         # ce_loss += F.cross_entropy(new_pred, new_y_onehot)

#                         loss += ins_loss + ce_loss
#                         loss_log['train/ins'] += ins_loss.item() if ins_loss != 0. else 0.
#                         loss_log['train/ce'] += ce_loss.item() if ce_loss != 0. else 0.

#                     else:
#                         # rotate and augment
#                         new_x = RandomFlip(new_x, 2)
#                         new_y = new_y.repeat(2)

#                         new_x = torch.cat((new_x, self.transform(new_x)))
#                         new_y = torch.cat((new_y, new_y))
#                         new_x = new_x.detach()
#                         new_y = new_y.detach()

#                         features = self.model.features(new_x)
#                         proj_feat = self.model.head(features, use_proj=True)
#                         pred_out = self.model.head(features, use_proj=False)
#                         ce_loss = F.cross_entropy(pred_out, new_y)

#                         ins_loss = sup_con_loss(proj_feat, self.ins_t, new_y)

#                         loss += ins_loss + ce_loss
#                         loss_log['train/ins'] += ins_loss.item() if ins_loss != 0. else 0.
#                         loss_log['train/ce'] += ce_loss.item() if ce_loss != 0. else 0.


#                 self.scaler.scale(loss).backward()
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
#                 self.optimizer.zero_grad()

#             if epoch == 0:
#                 self.buffer.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=task_id)

#             loss_log['train/loss'] = loss.item() if loss != 0. else 0.
#             epoch_log_holder.append(loss_log)
#             self.total_step += 1

#             if num_d % self.print_num == 0 or batch_idx == 1:
#                 print(f"==>>> it: {batch_idx}, loss: ins {loss_log['train/ins']:.2f} + ce {loss_log['train/ce']:.3f}, {100 * (num_d / self.total_samples)}%")

#         return epoch_log_holder

#     def train(self, task_id, train_loader):
#         self.pre_train(task_id)
#         self.model.train()
#         train_log_holder = []
#         for epoch in range(self.epoch):
#             epoch_log_holder = self.train_any_task(task_id, train_loader, epoch)
#             train_log_holder.extend(epoch_log_holder)
#             # self.buffer.print_per_task_num()
#         self.after_train(task_id)
#         return train_log_holder
    
#     def pre_train(self, task_id):
#         self.total_classes += self.class_per_task
#         new_classes_num = self.class_per_task if self.args.use_dummy_cls == 'off' else self.args.dummy_cls+self.class_per_task
#         self.model.add_head(new_classes_num, bias=True)
#         if self.reinit_optimizer == 'on':
#             self.optimizer = Adam(self.model.parameters(), self.args.lr, weight_decay=self.args.wd)  

#     def after_train(self, task_id):
#         self.known_classes += self.class_per_task
#         if self.args.use_dummy_cls == 'on' and self.args.dummy_cls > 0:
#             true_classes_nums = self.class_per_task
#             weights = self.model.multi_heads[task_id].weight.data
#             biases = self.model.multi_heads[task_id].bias.data
#             del self.model.multi_heads[task_id]
#             self.model.multi_heads.append(nn.Linear(self.model.out_dim, true_classes_nums, bias=True).cuda())
#             self.model.multi_heads[task_id].weight.data = weights[:true_classes_nums]
#             self.model.multi_heads[task_id].bias.data = biases[:true_classes_nums]

#     def test(self, i, task_loader):
#         self.model.eval()
#         if self.use_ncm:
#             # calculate the class means for each feature layer
#             print("\nCalculate class means for last layer...\n")
#             print(self.buffer.bx.shape)
#             self.class_means_ls = {}
#             class_inputs = {cls: [] for cls in self.class_holder}
#             for x, y in zip(self.buffer.x, self.buffer.y_int):
#                 class_inputs[y.item()].append(x)

#             for cls, inputs in class_inputs.items():
#                 features = []
#                 for ex in inputs:
#                     return_features = self.model.features(ex.unsqueeze(0))
#                     feature = return_features.detach().clone()
#                     feature = F.normalize(feature, dim=1)
#                     features.append(feature.squeeze())
                
#                 features = torch.stack(features)
#                 mu_y = features.mean(0)
#                 mu_y = F.normalize(mu_y.reshape(1, -1), dim=1)
#                 self.class_means_ls[cls] = mu_y.squeeze()

#         all_acc_list = {'step': self.total_step}

#         # test classifier from the last layer
#         print(f"{'*'*100}\nTest with the output of last layer:\n")
#         with torch.no_grad():
#             acc_list = np.zeros(len(task_loader))
#             for j in range(i + 1):
#                 acc = self.test_model(task_loader[j]['test'], j)
#                 acc_list[j] = acc.item()

#             all_acc_list['last'] = acc_list
#             print(f"tasks acc:{acc_list}")
#             print(f"tasks avg acc:{acc_list[:i+1].mean()}")

#         # # clear the calculated class_means
#         # self.class_means_ls = None

#         # test linear classifier
#         print(f"{'*'*100}\nTest with the output of linear classifier:\n")
#         with torch.no_grad():
#             acc_list = np.zeros(len(task_loader))
#             for j in range(i + 1):
#                 acc = self.test_model_linear(task_loader[j]['test'], j)
#                 acc_list[j] = acc.item()

#             all_acc_list['linear'] = acc_list
#             print(f"tasks acc:{acc_list}")
#             print(f"tasks avg acc:{acc_list[:i+1].mean()}")

#         return acc_list, all_acc_list
    
#     def test_model_linear(self, loader, i):
#         # test linear classifier
#         self.model.eval()
#         correct = torch.full([], 0).cuda()
#         num = torch.full([], 0).cuda()

#         for batch_idx, (x, y) in enumerate(loader):
#             x, y = x.cuda(), y.cuda()

#             features = self.model.features(x)
#             pred = self.model.head(features, use_proj=False)
#             pred = pred.data.max(1, keepdim=True)[1]

#             num += x.size()[0]
#             correct += pred.eq(y.data.view_as(pred)).sum()

#         test_accuracy = (100. * correct / num)
#         print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
#         return test_accuracy


#     def test_buffer(self, i, task_loader, feat_ids=[0,1,2,3]):
#         self.model.eval()
#         all_acc_list = {'step': self.total_step}
#         # test classifier from each required layer
#         for feat_id in feat_ids:
#             print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
#             with torch.no_grad():
#                 acc_list = np.zeros(len(task_loader))
#                 for j in range(i + 1):
#                     acc = self.test_buffer_task(j, feat_id=feat_id)
#                     acc_list[j] = acc.item()

#                 all_acc_list[str(feat_id)] = acc_list
#                 print(f"tasks acc:{acc_list}")
#                 print(f"tasks avg acc:{acc_list[:i+1].mean()}")

#         # test mean classifier
#         print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
#         with torch.no_grad():
#             acc_list = np.zeros(len(task_loader))
#             for j in range(i + 1):
#                 acc = self.test_buffer_task_mean(j)
#                 acc_list[j] = acc.item()

#             all_acc_list['mean'] = acc_list
#             print(f"tasks acc:{acc_list}")
#             print(f"tasks avg acc:{acc_list[:i+1].mean()}")

#         return acc_list, all_acc_list

#     def test_buffer_task(self, i, feat_id):
#         # test specific layer's output
#         correct = torch.full([], 0).cuda()
#         num = torch.full([], 0).cuda()

#         x_i, y_i, _ = self.buffer.onlysample(self.buffer.current_index, task=i)

#         if self.use_ncm:
#             class_means = self.class_means_ls[feat_id]
#             for x, y in zip(x_i, y_i):
#                 x = x.unsqueeze(0).detach()
#                 y = y.unsqueeze(0).detach()

#                 features = self.model.features(x)[feat_id]
#                 features = F.normalize(features, dim=1)
#                 features = features.unsqueeze(2)
#                 means = torch.stack([class_means[cls] for cls in self.class_holder])
#                 means = torch.stack([means] * x.size(0))
#                 means = means.transpose(1, 2)
#                 features = features.expand_as(means)
#                 dists = (features - means).pow(2).sum(1).squeeze(1)
#                 pred = dists.min(1)[1]
#                 pred = torch.Tensor(self.class_holder)[pred].to(x.device)

#                 num += x.size()[0]
#                 correct += pred.eq(y.data.view_as(pred)).sum()

#         else:
#             for x, y in zip(x_i, y_i):
#                 x = x.unsqueeze(0).detach()
#                 y = y.unsqueeze(0).detach()

#                 pred = self.model(x)[feat_id]
#                 pred = pred.data.max(1, keepdim=True)[1]

#                 num += x.size()[0]
#                 correct += pred.eq(y.data.view_as(pred)).sum()

#         test_accuracy = (100. * correct / num)
#         print('Buffer test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
#         return test_accuracy

#     def test_buffer_task_mean(self, i):
#         # test with mean dists for all layers
#         correct = torch.full([], 0).cuda()
#         num = torch.full([], 0).cuda()

#         x_i, y_i, _ = self.buffer.onlysample(self.buffer.current_index, task=i)

#         if self.use_ncm:
#             for x, y in zip(x_i, y_i):
#                 x = x.unsqueeze(0).detach()
#                 y = y.unsqueeze(0).detach()

#                 features_ls = self.model.features(x)
#                 dists_ls = []

#                 for feat_id in range(4):
#                     class_means = self.class_means_ls[feat_id]
#                     features = features_ls[feat_id]
#                     features = F.normalize(features, dim=1)
#                     features = features.unsqueeze(2)
#                     means = torch.stack([class_means[cls] for cls in self.class_holder])
#                     means = torch.stack([means] * x.size(0))
#                     means = means.transpose(1, 2)
#                     features = features.expand_as(means)
#                     dists = (features - means).pow(2).sum(1).squeeze(1)
#                     dists_ls.append(dists)

#                 dists_ls = torch.cat([dists.unsqueeze(1) for dists in dists_ls], dim=1)
#                 dists = dists_ls.mean(dim=1).squeeze(1)
#                 pred = dists.min(1)[1]
#                 pred = torch.Tensor(self.class_holder)[pred].to(x.device)

#                 num += x.size()[0]
#                 correct += pred.eq(y.data.view_as(pred)).sum()

#         else:
#             for x, y in zip(x_i, y_i):
#                 x = x.unsqueeze(0).detach()
#                 y = y.unsqueeze(0).detach()

#                 pred = self.model(x)
#                 pred = torch.stack(pred, dim=1)
#                 pred = pred.mean(dim=1).squeeze(1)
#                 pred = pred.data.max(1, keepdim=True)[1]

#                 num += x.size()[0]
#                 correct += pred.eq(y.data.view_as(pred)).sum()

#         test_accuracy = (100. * correct / num)
#         print('Buffer test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
#         return test_accuracy

#     def test_train(self, i, task_loader, feat_ids=[0,1,2,3]):
#         # train accuracy of current task i
#         self.model.eval()
#         all_acc_list = {'step': self.total_step}

#         # test classifier from each required layer
#         for feat_id in feat_ids:
#             print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
#             with torch.no_grad():
#                 acc_list = np.zeros(len(task_loader))
#                 acc = self.test_model(task_loader[i]['train'], i, feat_id=feat_id)
#                 acc_list[i] = acc.item()

#                 all_acc_list[str(feat_id)] = acc_list
#                 print(f"tasks acc:{acc_list}")
#                 print(f"tasks avg acc:{acc_list[:i+1].mean()}")

#         # test mean classifier
#         print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
#         with torch.no_grad():
#             acc_list = np.zeros(len(task_loader))
#             acc = self.test_model_mean(task_loader[i]['train'], i)
#             acc_list[i] = acc.item()

#             all_acc_list['mean'] = acc_list
#             print(f"tasks acc:{acc_list}")
#             print(f"tasks avg acc:{acc_list[:i+1].mean()}")

#         return acc_list, all_acc_list

#     def test_model(self, loader, i):
#         # test specific layer's output
#         correct = torch.full([], 0).cuda()
#         num = torch.full([], 0).cuda()
#         if self.use_ncm:
#             class_means = self.class_means_ls
#             class_holder_tensor = torch.tensor(self.class_holder).cuda()
#             for batch_idx, (data, target) in enumerate(loader):
#                 data, target = data.cuda(), target.cuda()

#                 features = self.model.features(data)
#                 features = F.normalize(features, dim=1)
#                 features = features.unsqueeze(2)
#                 means = torch.stack([class_means[cls] for cls in self.class_holder])
#                 means = torch.stack([means] * data.size(0))
#                 means = means.transpose(1, 2)
#                 features = features.expand_as(means)
#                 dists = (features - means).pow(2).sum(1).squeeze()
#                 pred = dists.min(1)[1]
#                 # pred = torch.Tensor(self.class_holder)[pred].to(data.device)

#                 pred = class_holder_tensor[pred]

#                 num += data.size()[0]
#                 correct += pred.eq(target.data.view_as(pred)).sum()

#         else:
#             for batch_idx, (data, target) in enumerate(loader):
#                 data, target = data.cuda(), target.cuda()

#                 pred = self.model(data)
#                 pred = pred.data.max(1, keepdim=True)[1]

#                 num += data.size()[0]
#                 correct += pred.eq(target.data.view_as(pred)).sum()

#         test_accuracy = (100. * correct / num)
#         print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
#         return test_accuracy

#     def test_model_mean(self, loader, i):
#         # test with mean dists for all layers
#         correct = torch.full([], 0).cuda()
#         num = torch.full([], 0).cuda()
#         if self.use_ncm:
#             class_holder_tensor = torch.tensor(self.class_holder).cuda()
#             for batch_idx, (data, target) in enumerate(loader):
#                 data, target = data.cuda(), target.cuda()
#                 features_ls = self.model.features(data)
#                 dists_ls = []

#                 for feat_id in range(4):
#                     class_means = self.class_means_ls[feat_id]
#                     features = features_ls[feat_id]
#                     features = F.normalize(features, dim=1)
#                     features = features.unsqueeze(2)
#                     means = torch.stack([class_means[cls] for cls in self.class_holder])
#                     means = torch.stack([means] * data.size(0))
#                     means = means.transpose(1, 2)
#                     features = features.expand_as(means)
#                     dists = (features - means).pow(2).sum(1).squeeze()
#                     dists_ls.append(dists)

#                 dists_ls = torch.cat([dists.unsqueeze(1) for dists in dists_ls], dim=1)
#                 dists = dists_ls.mean(dim=1).squeeze(1)
#                 pred = dists.min(1)[1]
#                 # pred = torch.Tensor(self.class_holder)[pred].to(data.device)
#                 pred = class_holder_tensor[pred]

#                 num += data.size()[0]
#                 correct += pred.eq(target.data.view_as(pred)).sum()

#         else:
#             for batch_idx, (data, target) in enumerate(loader):
#                 data, target = data.cuda(), target.cuda()

#                 pred = self.model(data)
#                 pred = torch.stack(pred, dim=1)
#                 pred = pred.mean(dim=1).squeeze()
#                 pred = pred.data.max(1, keepdim=True)[1]

#                 num += data.size()[0]
#                 correct += pred.eq(target.data.view_as(pred)).sum()

#         test_accuracy = (100. * correct / num)
#         print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
#         return test_accuracy

#     def save_checkpoint(self, save_path = './outputs/final.pt'):
#         print(f"Save checkpoint to: {save_path}")
#         ckpt_dict = {
#             'model': self.model.state_dict(),
#             'buffer': self.buffer.state_dict(),
#         }
#         folder, file_name = os.path.split(save_path)
#         if not os.path.isdir(folder):
#             os.mkdir(folder)
#         torch.save(ckpt_dict, save_path)

#     def load_checkpoint(self, load_path = './outputs/final.pt'):
#         print(f"Load checkpoint from: {load_path}")
#         ckpt_dict = torch.load(load_path)
#         self.model.load_state_dict(ckpt_dict['model'])
#         self.buffer.load_state_dict(ckpt_dict['buffer'])




import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast

from losses.loss import sup_con_loss
from utils import get_transform
from utils.rotation_transform import RandomFlip
from models.Resnet_cifar import resnet18_cifar, resnet34_cifar, resnet50_cifar
from losses.distill_loss import DistillKL


class Dist(object):
    def __init__(self, model:nn.Module, buffer, optimizer, input_size, args):
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

        if args.dataset == "cifar10":
            self.total_samples = 10000
        elif "cifar100" in args.dataset:
            self.total_samples = 5000
        elif args.dataset == "tiny_imagenet":
            self.total_samples = 1000
        self.print_num = self.total_samples // 10

        self.transform = get_transform(args.augmentation, input_size)

        self.total_step = 0
        self.class_holder = []
        self.scaler = GradScaler()
        self.args = args
        self.class_per_task = args.class_per_task
        self.known_classes = 0
        self.total_classes = 0
        self.reinit_optimizer = args.reinit_optimizer

    def train_any_task(self, task_id, train_loader, epoch):
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
            loss_log = {
                'step':     self.total_step,
                'train/loss':     0.,
                'train/ins':      0.,
                'train/ce':       0.,
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
                        buffer_cur_task = self.buffer_batch_size if task_id==0 else self.buffer_cur_task
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

                        new_x = torch.cat((new_x, self.transform(new_x)))
                        new_y = torch.cat((new_y, new_y))
                        cat_x = torch.cat((cat_x, self.transform(cat_x)))
                        cat_y = torch.cat((cat_y, cat_y))

                        new_input_size = new_x.size(0)
                        cat_input_size = cat_x.size(0)

                        all_x = torch.cat((new_x, cat_x))
                        all_y = torch.cat((new_y, cat_y))
                        all_x = all_x.detach()
                        all_y = all_y.detach()

                        features = self.model.features(all_x)
                        proj_feat = self.model.head(features, use_proj=True)

                        new_feat = features[:new_input_size]
                        cat_feat = features[new_input_size:]

                        cat_pred = self.model.head(cat_feat, use_proj=False)
                        new_pred = self.model.multi_heads[task_id](new_feat)

                        ins_loss = sup_con_loss(proj_feat, self.ins_t, all_y)

                        ce_loss = 2 * F.cross_entropy(cat_pred[:, :self.total_classes], cat_y)

                        fake_targets = new_y - self.known_classes
                        ce_loss += F.cross_entropy(new_pred, fake_targets)

                        loss += ins_loss + ce_loss
                        loss_log['train/ins'] += ins_loss.item() if ins_loss != 0. else 0.
                        loss_log['train/ce'] += ce_loss.item() if ce_loss != 0. else 0.

                    else:
                        # rotate and augment
                        new_x = RandomFlip(new_x, 2)
                        new_y = new_y.repeat(2)

                        new_x = torch.cat((new_x, self.transform(new_x)))
                        new_y = torch.cat((new_y, new_y))
                        new_x = new_x.detach()
                        new_y = new_y.detach()

                        features = self.model.features(new_x)
                        proj_feat = self.model.head(features, use_proj=True)
                        pred = self.model.multi_heads[task_id](features)

                        ins_loss = sup_con_loss(proj_feat, self.ins_t, new_y)
                        ce_loss = F.cross_entropy(pred, new_y)

                        loss += ins_loss + ce_loss
                        loss_log['train/ins'] += ins_loss.item() if ins_loss != 0. else 0.
                        loss_log['train/ce'] += ce_loss.item() if ce_loss != 0. else 0.

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
                print(f"==>>> it: {batch_idx}, loss: ins {loss_log['train/ins']:.2f} + ce {loss_log['train/ce']:.3f}, {100 * (num_d / self.total_samples)}%")

        return epoch_log_holder

    def train(self, task_id, train_loader):
        self.pre_train(task_id)
        self.model.train() 
        train_log_holder = []
        for epoch in range(self.epoch):
            epoch_log_holder = self.train_any_task(task_id, train_loader, epoch)
            train_log_holder.extend(epoch_log_holder)
            # self.buffer.print_per_task_num()
        self.after_train(task_id)
        return train_log_holder
    
    def pre_train(self, task_id):
        self.total_classes += self.class_per_task
        if self.args.use_dummy_cls == 'on' and task_id > 0:
            new_classes_num = self.args.dummy_cls + self.class_per_task
        else:
            new_classes_num = self.class_per_task
        self.model.add_head(new_classes_num, bias=self.args.fc_bias)
        if self.reinit_optimizer == 'on':
            self.optimizer = Adam(self.model.parameters(), self.args.lr, weight_decay=self.args.wd)

    def after_train(self, task_id):
        self.known_classes += self.class_per_task
        if self.args.use_dummy_cls == 'on' and self.args.dummy_cls > 0:
            true_classes_nums = self.class_per_task
            weights = self.model.multi_heads[task_id].weight.data
            biases = self.model.multi_heads[task_id].bias.data
            del self.model.multi_heads[task_id]
            self.model.multi_heads.append(nn.Linear(self.model.out_dim, true_classes_nums, bias=self.args.fc_bias).cuda())
            self.model.multi_heads[task_id].weight.data = weights[:true_classes_nums]
            self.model.multi_heads[task_id].bias.data = biases[:true_classes_nums]
        torch.save(self.model.state_dict(), os.path.join(self.args.log_path, f"model_{task_id}.pth"))

    def test(self, i, task_loader):
        self.model.eval()
        if self.use_ncm:
            # calculate the class means for each feature layer
            print("\nCalculate class means for last layer...\n")
            print(self.buffer.bx.shape)
            self.class_means_ls = {}
            class_inputs = {cls: [] for cls in self.class_holder}
            for x, y in zip(self.buffer.x, self.buffer.y_int):
                class_inputs[y.item()].append(x)

            for cls, inputs in class_inputs.items():
                features = []
                for ex in inputs:
                    return_features = self.model.features(ex.unsqueeze(0))
                    feature = return_features.detach().clone()
                    feature = F.normalize(feature, dim=1)
                    features.append(feature.squeeze())
                
                features = torch.stack(features)
                mu_y = features.mean(0)
                mu_y = F.normalize(mu_y.reshape(1, -1), dim=1)
                self.class_means_ls[cls] = mu_y.squeeze()

        all_acc_list = {'step': self.total_step}

        # test classifier from the last layer
        print(f"{'*'*100}\nTest with the output of last layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model(task_loader[j]['test'], j)
                acc_list[j] = acc.item()

            all_acc_list['last'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # # clear the calculated class_means
        # self.class_means_ls = None

        # test linear classifier
        print(f"{'*'*100}\nTest with the output of linear classifier:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_model_linear(task_loader[j]['test'], j)
                acc_list[j] = acc.item()

            all_acc_list['linear'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

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


    def test_buffer(self, i, task_loader, feat_ids=[0,1,2,3]):
        self.model.eval()
        all_acc_list = {'step': self.total_step}
        # test classifier from each required layer
        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                for j in range(i + 1):
                    acc = self.test_buffer_task(j, feat_id=feat_id)
                    acc_list[j] = acc.item()

                all_acc_list[str(feat_id)] = acc_list
                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # test mean classifier
        print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                acc = self.test_buffer_task_mean(j)
                acc_list[j] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

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

    def test_train(self, i, task_loader, feat_ids=[0,1,2,3]):
        # train accuracy of current task i
        self.model.eval()
        all_acc_list = {'step': self.total_step}

        # test classifier from each required layer
        for feat_id in feat_ids:
            print(f"{'*'*100}\nTest with the output of layer: {feat_id+1}\n")
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                acc = self.test_model(task_loader[i]['train'], i, feat_id=feat_id)
                acc_list[i] = acc.item()

                all_acc_list[str(feat_id)] = acc_list
                print(f"tasks acc:{acc_list}")
                print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        # test mean classifier
        print(f"{'*'*100}\nTest with the mean dists output of each layer:\n")
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            acc = self.test_model_mean(task_loader[i]['train'], i)
            acc_list[i] = acc.item()

            all_acc_list['mean'] = acc_list
            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i+1].mean()}")

        return acc_list, all_acc_list

    def test_model(self, loader, i):
        # test specific layer's output
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        if self.use_ncm:
            class_means = self.class_means_ls
            class_holder_tensor = torch.tensor(self.class_holder).cuda()
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()

                features = self.model.features(data)
                features = F.normalize(features, dim=1)
                features = features.unsqueeze(2)
                means = torch.stack([class_means[cls] for cls in self.class_holder])
                means = torch.stack([means] * data.size(0))
                means = means.transpose(1, 2)
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

    def test_model_mean(self, loader, i):
        # test with mean dists for all layers
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        if self.use_ncm:
            class_holder_tensor = torch.tensor(self.class_holder).cuda()
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.cuda(), target.cuda()
                features_ls = self.model.features(data)
                dists_ls = []

                for feat_id in range(4):
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

    def save_checkpoint(self, save_path = './outputs/final.pt'):
        print(f"Save checkpoint to: {save_path}")
        ckpt_dict = {
            'model': self.model.state_dict(),
            'buffer': self.buffer.state_dict(),
        }
        folder, file_name = os.path.split(save_path)
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save(ckpt_dict, save_path)

    def load_checkpoint(self, load_path = './outputs/final.pt'):
        print(f"Load checkpoint from: {load_path}")
        ckpt_dict = torch.load(load_path)
        self.model.load_state_dict(ckpt_dict['model'])
        self.buffer.load_state_dict(ckpt_dict['buffer'])
