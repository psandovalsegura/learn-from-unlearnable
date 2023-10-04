import os
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
import argparse
import torchvision.transforms as transforms
from tqdm import tqdm
from utils_projection import project_data, CustomTensorDataset    
from datasets import construct_train_dataset, get_test_dataset
from utils_scheduler import GradualWarmupScheduler
from utils_models import get_model
from constants import LINEAR_CKPT_DIR, DATA_SETUPS, NUM_CLASSES, TRANSFORM_OPTIONS

class LinearModel(torch.nn.Module):
    def __init__(self, image_shape, num_classes=10):
        super(LinearModel, self).__init__()
        input_dim = np.prod(image_shape)
        self.fc = torch.nn.Linear(input_dim, num_classes, bias=False)
        
    def forward(self, x):
        x = self.fc(x.view((x.shape[0], -1)))
        return x
    
    def project_weights(self, max_val=1.0):
        self.fc.weight.data = torch.clamp(self.fc.weight.data, -max_val, max_val)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch, args):
    # switch to train mode
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.train()
    for i, (input, target) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}"):
        input = input.cuda()
        target = target.long().cuda()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)
            
        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"train_loss: {losses.avg} train_acc: {100-top1.avg}")
    return losses.avg


def validate(val_loader, model, criterion, epoch, args):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.long().cuda()
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
    print(f"val_loss: {losses.avg} val_acc: {100-top1.avg}")
    return losses.avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('poison_identifier', type=str)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--momentum', type=int, default=0.9)
    parser.add_argument('--weight_decay', type=int, default=1e-4)
    parser.add_argument('--beta', default=1, type=float,
                        help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=0.5, type=float,
                        help='cutmix probability')
    parser.add_argument('--no-project', action='store_true')
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--no-strong-aug', action='store_true')
    parser.add_argument('--project-test', action='store_true')
    return parser.parse_args()


def main(args):
    dataset = construct_train_dataset(args.poison_identifier, normalize=False, transforms_key='test_transform', is_ortho_proj=True)
    print("loaded training data")

    image_shape = dataset[0][0].shape
    dataset_name = DATA_SETUPS[args.poison_identifier]['dataset_name']
    num_classes = NUM_CLASSES[dataset_name]
    linear_model = LinearModel(image_shape, num_classes=num_classes).cuda()
    linear_model_ckpt_path = os.path.join(LINEAR_CKPT_DIR, args.poison_identifier+'.pt')
    linear_model.load_state_dict(torch.load(linear_model_ckpt_path))
    print("loaded linear model checkpoint from", linear_model_ckpt_path)
        
    data = np.stack([x[0].view(-1).numpy() for x in dataset])
    label = [x[1] for x in dataset]

    linear_coef = linear_model.fc.weight.data.cpu().numpy()
    if args.no_project:
        image_data = data.reshape(-1, *image_shape)
    else:
        print('projecting data')
        projected_data = project_data(data, linear_coef, True)
        image_data = projected_data.reshape(-1, *image_shape)

    if not args.no_strong_aug:
        print('using strong augmentations')
        image_side_len = image_shape[1]
        if dataset_name in ['CIFAR10', 'CIFAR100', 'SVHN']:
            normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #Normalize all the images
        elif dataset_name == 'IMAGENET100':
            normalize_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_transform = transforms.Compose([transforms.RandomRotation(10),     #Rotates the image to a specified angle
                                            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                            transforms.RandomCrop(image_side_len, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            normalize_transform
                                            ])
    else:
        print('not using strong augmentations')
        train_transform = transforms.Compose(TRANSFORM_OPTIONS[dataset_name]['train_transform'][:-1]) # drop ToTensor transform from list
    tensor_dataset = CustomTensorDataset((torch.Tensor(image_data), torch.Tensor(label)),
                                         transform=train_transform)    
    trainloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    testloader = get_test_dataset(args.poison_identifier, args.batch_size, num_workers=args.num_workers, normalize=True)
    
    model = get_model(args.model, args.poison_identifier)
    EPOCHS = args.epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Ortho Proj training: choose scheduler for rn18 based on dataset
    if dataset_name in ['CIFAR10', 'SVHN', 'CIFAR100']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EPOCHS*0.5),int(EPOCHS*0.75)], gamma=0.1)
    elif dataset_name in ['IMAGENET100']: # has not been tested (as of 5/23/23)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EPOCHS*0.5),int(EPOCHS*0.75), int(EPOCHS*0.90)], gamma=0.1)
    else:
        raise ValueError(f"Dataset {dataset_name} not yet supported for ortho proj training, scheduler should be tuned for different number of classes")
    if 'vit' in args.model:
        # Use Adam optimizer with cosine annealing LR scheduler for ViT models
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-5)
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler=base_scheduler)
    
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    for epoch in range(EPOCHS):
        train(trainloader, model, criterion, optimizer, epoch, args)
        scheduler.step()
        validate(testloader, model, criterion, epoch, args)

if __name__=='__main__':
    args = parse_args()
    main(args)