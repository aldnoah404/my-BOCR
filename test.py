from models.resnet56 import pruned_resnet, P_BasicBlock
import torch
from lib.data import get_split_dataset
from lib.utils import AverageMeter, accuracy, prGreen
import torch.nn as nn


# 加载剪枝后模型
ratios = [3, 8, 9, 9, 7, 9, 8, 7, 8, 9, 17, 14, 14, 17, 17, 14, 14, 17, 14, 33, 29, 33, 35, 35, 29, 29, 35, 32, 64]
model = pruned_resnet(P_BasicBlock, [9, 9, 9], ratios)

# 加载预训练模型
model.load_state_dict(torch.load('./purned_model/resnet56_static.pth', weights_only=True))
model = model.cuda()

# 加载数据集
train_loader, val_loader, n_class = get_split_dataset('cifar10', 50, 6, 5000, data_root='./dataset', use_real_val=False, shuffle=True)

losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

criterion = nn.CrossEntropyLoss().cuda()

model.eval()
with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print('* Test loss: %.3f    top1: %.3f    top5: %.3f ' % (losses.avg, top1.avg, top5.avg))

