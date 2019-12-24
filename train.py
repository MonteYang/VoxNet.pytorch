# coding: utf-8
from __future__ import print_function
import argparse
import sys
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from voxnet import VoxNet
sys.path.insert(0, './data/')
from modelnet10 import ModelNet10

CLASSES = {
    0: 'bathtub',
    1: 'chair',
    2: 'dresser',
    3: 'night_stand',
    4: 'sofa',
    5: 'toilet',
    6: 'bed',
    7: 'desk',
    8: 'monitor',
    9: 'table'
}
N_CLASSES = len(CLASSES)

def blue(x): return '\033[94m' + x + '\033[0m'

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='/Data1/DL-project/VoxNet.pytorch/data/ModelNet10', help="dataset path")
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--n-epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
opt = parser.parse_args()
# print(opt)

# 创建目录
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# 固定随机种子
opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 数据加载
train_dataset = ModelNet10(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='train')
test_dataset = ModelNet10(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='test')

train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

# VoxNet
voxnet = VoxNet(n_classes=N_CLASSES)

print(voxnet)

# 加载权重
if opt.model != '':
    voxnet.load_state_dict(torch.load(opt.model))

# 优化器
optimizer = optim.Adam(voxnet.parameters(), lr=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
voxnet.cuda()

num_batch = len(train_dataset) / opt.batchSize
print(num_batch)

for epoch in range(opt.n_epoch):
    # scheduler.step()
    for i, sample in enumerate(train_dataloader, 0):
        # 读数据
        voxel, cls_idx = sample['voxel'], sample['cls_idx']
        voxel, cls_idx = voxel.cuda(), cls_idx.cuda()
        voxel = voxel.float()  # Voxel原来是int类型(0,1),需转float, torch.Size([256, 1, 32, 32, 32])

        # 梯度清零
        optimizer.zero_grad()

        # 网络切换训练模型
        voxnet = voxnet.train()
        pred = voxnet(voxel)  # torch.Size([256, 10])

        # 计算损失函数
        
        loss = F.cross_entropy(pred, cls_idx)

        # 反向传播, 更新权重
        loss.backward()
        optimizer.step()

        # 计算该batch的预测准确率
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(cls_idx.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %
              (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        # 每5个batch进行一次test
        if i % 5 == 0:
            j, sample = next(enumerate(test_dataloader, 0))
            voxel, cls_idx = sample['voxel'], sample['cls_idx']
            voxel, cls_idx = voxel.cuda(), cls_idx.cuda()
            voxel = voxel.float()  # 转float, torch.Size([256, 1, 32, 32, 32])
            voxnet = voxnet.eval()
            pred = voxnet(voxel)
            loss = F.nll_loss(pred, cls_idx)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(cls_idx.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch,
                                                            blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    # 保存权重
    torch.save(voxnet.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))


# 训练后, 在测试集上评估
total_correct = 0
total_testset = 0

for i, data in tqdm(enumerate(test_dataloader, 0)):
    voxel, cls_idx = data['voxel'], data['cls_idx']
    voxel, cls_idx = voxel.cuda(), cls_idx.cuda()
    voxel = voxel.float()  # 转float, torch.Size([256, 1, 32, 32, 32])

    voxnet = voxnet.eval()
    pred = voxnet(voxel)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(cls_idx.data).cpu().sum()
    total_correct += correct.item()
    total_testset += voxel.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
