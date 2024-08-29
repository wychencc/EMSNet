import argparse
import time
import os
import json
from dataset import OurDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from libs import average_meter, metric
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR
from models.my import My
from tensorboardX import SummaryWriter
import sync_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Remote Sensing Segmentation by PyTorch")
    # dataset
    parser.add_argument('--dataset-name', type=str, default='six')

    # -===================！！！！！！！
    parser.add_argument('--train-data-root', type=str, default='XXXX/train/')
    parser.add_argument('--val-data-root', type=str, default='XXXX/val/')
    parser.add_argument('--save_root', type=str, default='XXXX')
    parser.add_argument('--gpu_ids', type=list, default=[0])
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='weight-decay (default:1e-4)')

    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='batch size for training (default:16)')
    parser.add_argument('--val-batch-size', type=int, default=16, metavar='N',
                        help='batch size for testing (default:16)')
    # output_save_path 
    parser.add_argument('--experiment-start-time', type=str,
                        default=time.strftime('%m-%d-%H:%M:%S', time.localtime(time.time())))

    # learning_rate 
    parser.add_argument('--base-lr', type=float, default=1e-3, metavar='M', help='')

    parser.add_argument('--total-epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 60)')

    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--base-size', type=int, default=256, help='base image size')
    parser.add_argument('--crop-size', type=int, default=256, help='crop image size')
    parser.add_argument('--flip-ratio', type=float, default=0.5)
    parser.add_argument('--resize-scale-range', type=str, default='0.5, 2.0')
    # -===================！！！！！！！
    parser.add_argument('--model', type=str, default='my', help='model name')
    parser.add_argument('--class-loss-weight', type=list,
                        default=[6.07649307,
                                 1.0,
                                 1.15991088,
                                 1.30049189,
                                 1.87984675,
                                 16.59994979])

    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='start epoch (default:0)')

    # loss
    parser.add_argument('--loss-names', type=str, default='cross_entropy')
    parser.add_argument('--classes-weight', type=str, default=None)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default:0.9)')

    # optimizer
    parser.add_argument('--optimizer-name', type=str, default='Adam')

    # environment
    parser.add_argument('--use-cuda', action='store_true', default=True, help='using CUDA training')
    parser.add_argument('--num-GPUs', type=int, default=2, help='numbers of GPUs')

    parser.add_argument('--num_workers', type=int, default=4)
    # validation
    parser.add_argument('--eval', action='store_true', default=False, help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False)

    parser.add_argument('--best-kappa', type=float, default=0)

    parser.add_argument('--resume-path', type=str, default=None)

    parser.add_argument('--resume_model', type=bool, default=False)
    parser.add_argument('--resume_model_path', type=str, default='')
    parser.add_argument('--resume_start_epoch', type=int, default=0)
    parser.add_argument('--resume_total_epochs', type=int, default=500)
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()
    directory = args.save_root + "/%s/" % (args.experiment_start_time)
    args.directory = directory

    if not os.path.exists(directory):
        os.makedirs(directory)
    config_file = os.path.join(directory, 'config.json')
    with open(config_file, 'w') as file:
        json.dump(vars(args), file, indent=4)

    if args.use_cuda:
        print('Numbers of GPUs:', len(args.gpu_ids))
    else:
        print("Using CPU")
    return args


class DeNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Trainer(object):
    def __init__(self, args):
        self.args = args
        sync_transform = sync_transforms.ComposeWHU([
            sync_transforms.RandomFlipWHU(args.flip_ratio)
        ])
        self.resore_transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        self.visualize = transforms.Compose([transforms.ToTensor()])  # /255.

        dataset_name = args.dataset_name
        class_name = []
        if dataset_name == 'fifteen':
            from class_names import fifteen_classes
            class_name = fifteen_classes()
        if dataset_name == 'eight':
            from class_names import eight_classes
            class_name = eight_classes()
        if dataset_name == 'five':
            from class_names import five_classes
            class_name = five_classes()
        if dataset_name == 'seven':
            from class_names import seven_classes
            class_name = seven_classes()
        if dataset_name == 'six':
            from class_names import six_classes
            class_name = six_classes()
        self.train_dataset = OurDataset(class_name, root=args.train_data_root, mode='train',
                                              sync_transforms=sync_transform)  # random flip
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=args.train_batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=True,
                                       drop_last=True)

        print('class names {}.'.format(self.train_dataset.class_names))
        print('Number samples {}.'.format(len(self.train_dataset)))
        if not args.no_val:
            val_data_set = OurDataset(class_name, root=args.val_data_root, mode='val', sync_transforms=None)
            self.val_loader = DataLoader(dataset=val_data_set,
                                         batch_size=args.val_batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=False,
                                         drop_last=True)
        self.num_classes = len(self.train_dataset.class_names)
        print("类别数：", self.num_classes)  # 16
        print(self.train_dataset.class_names)
        self.class_loss_weight = torch.Tensor(args.class_loss_weight)
        # -===================！！！！！！！  ignore 0
        self.criterion_cross = nn.CrossEntropyLoss(weight=self.class_loss_weight, reduction='mean',
                                                   ignore_index=0).cuda()

        if args.model == 'my':
            model = My(num_classes=6)
            print('======> model My (Paper) =============== ')

        if args.resume_model:
            print('resume model', args.resume_model)
            state_dict = torch.load(args.resume_model_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print('=========> resume model success', args.resume_model_path)

        if args.use_cuda:
            model = model.cuda()
            # self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            # -===================！！！！！！！  
            self.model = nn.DataParallel(model, device_ids=args.gpu_ids)

        # SGD不work，Adadelta出奇的好？
        if args.optimizer_name == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(model.parameters(),
                                                  lr=args.base_lr,
                                                  weight_decay=args.weight_decay)
        if args.optimizer_name == 'Adam':
            # -===================！！！！！！！  ignore 0
            self.optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),
                                              lr=args.base_lr, weight_decay=args.weight_decay)
        if args.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(params=model.parameters(),
                                             lr=args.base_lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weight_decay)

        self.max_iter = args.total_epochs * len(self.train_loader)

    def training(self, epoch):
        self.model.train()
        # 把module设成训练模式，对Dropout和BatchNorm有影响
        train_loss = average_meter.AverageMeter()
        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.train_loader)
        for index, data in enumerate(tbar):

            imgs_sar = Variable(data[0])
            imgs_opt = Variable(data[1])
            masks = Variable(data[2])

            if self.args.use_cuda:
                imgs_sar = imgs_sar.cuda()
                imgs_opt = imgs_opt.cuda()
                masks = masks.cuda()

            self.optimizer.zero_grad()

            outputs = self.model(imgs_sar, imgs_opt)
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)

            loss = self.criterion_cross(outputs, masks)
            train_loss.update(loss, self.args.train_batch_size)
            loss.backward()
            self.optimizer.step()

            tbar.set_description(
                'epoch {}/{}, training loss {}, with learning rate {}.'.format(epoch, args.total_epochs, train_loss.avg,
                                                                               self.optimizer.state_dict()[
                                                                                   'param_groups'][0]['lr']))

            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)

            conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                label=masks.flatten(),
                                                num_classes=self.num_classes)

        train_acc, train_acc_per_class, train_acc_cls, train_IoU, train_mean_IoU, train_kappa = metric.evaluate(
            conf_mat)
        for i in range(self.num_classes - 1):
            print('====> class id ', i, self.train_dataset.class_names[i + 1], train_acc_per_class[i], train_IoU[i])
        # print(table)
        print("train_acc (OA):", train_acc)
        print("train_mean_IoU (Iou):", train_mean_IoU)
        print("kappa (Kappa):", train_kappa)

    def validating(self, epoch):
        self.model.eval()  # 把module设成预测模式，对Dropout和BatchNorm有影响
        conf_mat = np.zeros((self.num_classes, self.num_classes)).astype(np.int64)
        tbar = tqdm(self.val_loader)
        for index, data in enumerate(tbar):

            imgs_sar = Variable(data[0])
            imgs_opt = Variable(data[1])
            masks = Variable(data[2])

            if self.args.use_cuda:
                imgs_sar = imgs_sar.cuda()
                imgs_opt = imgs_opt.cuda()
                masks = masks.cuda()

            self.optimizer.zero_grad()


            outputs = self.model(imgs_sar, imgs_opt)
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
            masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)

            conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                label=masks.flatten(),
                                                num_classes=self.num_classes)

        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = metric.evaluate(conf_mat)

        model_name = 'epoch_%d_oa_%.5f_kappa_%.5f' % (epoch, val_acc, val_kappa)
        if val_kappa > self.args.best_kappa:
            torch.save(self.model.state_dict(), os.path.join(self.args.directory, model_name + '.pth'))
            self.args.best_kappa = val_kappa

        torch.save(self.model.state_dict(),
                   os.path.join(self.args.directory, model_name + '_latest.pth'))  # arg.directory changed

        for i in range(self.num_classes - 1):
            print('====> class id ', i, self.train_dataset.class_names[i + 1], val_acc_per_class[i], val_IoU[i])
        # print(table)
        print("val_acc (OA):", val_acc)
        print("val_mean_IoU (Iou):", val_mean_IoU)
        print("kappa (Kappa):", val_kappa)


if __name__ == "__main__":
    args = parse_args()
    writer = SummaryWriter(args.directory)
    trainer = Trainer(args)

    if args.eval:
        # print("Evaluating model:", args.resume)
        trainer.validating(epoch=0)
    else:
        print("Starting Epoch:", args.start_epoch)

    if args.resume_model:
        print("=====> Continue Train:")
        args.start_epoch = args.resume_start_epoch
        args.total_epochs = args.resume_total_epochs
    scheduler = StepLR(trainer.optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(args.start_epoch, args.total_epochs):
        trainer.training(epoch)
        scheduler.step()
        if not trainer.args.no_val:
            trainer.validating(epoch)
