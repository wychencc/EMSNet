import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from collections import OrderedDict
import torch.nn as nn
from dataset import OurDataset
from dataset import img_sar_transform, img_opt_transform, mask_transform
from torchvision import transforms
from libs import average_meter, metric
from models.my import My

img_transform = transforms.Compose([
    transforms.ToTensor()])

resore_transform = transforms.Compose([
    transforms.ToPILImage()
])

from class_names import eight_classes, two_classes

class_name = two_classes()


def snapshot_forward(model, dataloader, save_path, num_classes, args):
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes)).astype(np.int64)
    for index, data in enumerate(dataloader):

        imgs_sar = Variable(data[0])
        imgs_opt = Variable(data[1])
        masks = Variable(data[2])

        imgs_sar = imgs_sar.cuda()
        imgs_opt = imgs_opt.cuda()
        masks = masks.cuda()

        with torch.no_grad():
            torch.cuda.synchronize()
            outputs = model(imgs_sar, imgs_opt)
            torch.cuda.synchronize()

        preds = torch.argmax(outputs, 1)
        preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
        masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)

        for i in range(masks.shape[0]):
            conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                label=masks.flatten(),
                                                num_classes=num_classes)

    test_acc, test_acc_per_class, test_acc_cls, test_IoU, test_mean_IoU, test_kappa = metric.evaluate(conf_mat)
    print("test_acc:", test_acc)
    print("test_acc_cls:", test_acc_cls)
    print("test_mean_IoU:", test_mean_IoU)
    print("test kappa:", test_kappa)
    for i in range(num_classes):
        print(i, two_classes()[i], test_acc_per_class[i], test_IoU[i])


def parse_args():
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument('--test-data-root', type=str, default=r'XXXXX/test/')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='batch size for testing (default:16)')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument("--model-path", type=str,
                        default=r"XXXX/XXX.pth")
    parser.add_argument("--pred-path", type=str, default="")
    args = parser.parse_args()
    return args

def reference():
    args = parse_args()
    dataset = OurDataset(class_name=class_name,
                         root=args.test_data_root,
                         img_sar_transform=img_sar_transform, img_opt_transform=img_opt_transform,
                         mask_transform=mask_transform,
                         sync_transforms=None
                         )

    dataloader = DataLoader(dataset=dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    print(len(dataloader))
    print(class_name, len(class_name))

    model = My(num_classes=2)

    state_dict = torch.load(args.model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print('=========> load model success', args.model_path)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])

    snapshot_forward(model, dataloader, args.pred_path, len(class_name), args)
    print('test done........')


if __name__ == '__main__':
    reference()
