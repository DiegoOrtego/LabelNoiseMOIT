
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
import argparse
import logging
import os
import time

from dataset.cifar10_dataset import *
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models

import random
import sys

sys.path.append('../utils_labelNoise')
from utils_noise import *

from utils.criterion import accuracy_v2
from utils.AverageMeter import AverageMeter
import models as mod


logger = logging.getLogger()
logger.setLevel(logging.INFO)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    parser.add_argument('--epoch', type=int, default=200, help='training epoches')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of in-distribution classes')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--noise_type', default='asymmetric', help='noise type of the dataset')
    parser.add_argument('--train_root', default='./data', help='root for train data')
    parser.add_argument('--noise_ratio', type=float, default=0.5, help='percent of noise')
    parser.add_argument('--out', type=str, default='./data/model_data', help='Directory of the output')
    parser.add_argument('--alpha', type=float, default=1.0, help='Beta distribution parameter for mixup')
    parser.add_argument('--download', type=bool, default=False, help='download dataset')
    parser.add_argument('--network', type=str, default='PR18', help='Network architecture')
    parser.add_argument('--seed_initialization', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_dataset', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='name of the experiment (for the output files)')
    parser.add_argument('--method', type=str, default='NegativeLearning', help='None (Cross-entropy), Mixup, HardDynamicBoot. SoftDynamicBoot, SoftRelabeling')
    parser.add_argument('--dataset', type=str, default='CIFAR-100', help='CIFAR-10, CIFAR-100')
    parser.add_argument('--initial_epoch', type=int, default=1, help="Star training at initial_epoch")
    parser.add_argument('--DA', type=str, default="simple", help='Choose simple or complex data augmentation')
    parser.add_argument('--low_dim', type=int, default=128, help='Size of contrastive learning embedding')
    parser.add_argument('--batch_t', default=0.1, type=float, help='Contrastive learning temperature')
    parser.add_argument('--aprox', type=int, default=1, help='Warm-up epochs')
    parser.add_argument('--headType', type=str, default="Linear", help='Linear, NonLinear')
    parser.add_argument('--xbm_use', type=int, default=1, help='1: Use xbm')
    parser.add_argument('--xbm_begin', type=int, default=1, help='Epoch to begin using memory')
    parser.add_argument('--xbm_per_class', type=int, default=20, help='Num of samples per class to store in the memory. Memory size = xbm_per_class*num_classes')
    parser.add_argument('--k_val', type=int, default=5, help='k for k-nn correction')
    parser.add_argument('--validation_exp', type=int, default=0, help='Using clean train subset for validation')
    parser.add_argument('--val_samples', type=int, default=5000, help='Number of samples used for validation')
    parser.add_argument('--startLabelCorrection', type=int, default=9999, help='Epoch to start label correction')
    parser.add_argument('--PredictiveCorrection', type=int, default=0, help='Enable predictive label correction')
    parser.add_argument('--ReInitializeClassif', type=int, default=0, help='Enable predictive label correction')

    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test, clean_idx):

    trainset, testset, clean_labels, noisy_labels, noisy_indexes, all_labels = get_dataset(args, transform_train, transform_test)

    ## Get only detected clean samples
    trainset.data = trainset.data[clean_idx==1]
    trainset.targets = trainset.targets[clean_idx==1]

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print('############# Data loaded #############')

    return train_loader, test_loader, clean_labels, noisy_labels, noisy_indexes, trainset, all_labels



def main(args):
    # best_ac only record the best top1_ac for validation set.
    best_ac = 0.0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed_initialization)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed_initialization)  # GPU seed

    random.seed(args.seed_initialization)  # python seed for image transformation

    if args.dataset == 'CIFAR-10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'CIFAR-100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # data loader
    num_classes = args.num_classes

    clean_idx = np.load(
        './metrics' + args.network + "_Test_MOIT_" + args.noise_type + "_SI" + str(
            args.seed_initialization) + "_" + "SD" + str(args.seed_dataset) +
        "/" + str(args.noise_ratio) + "/agreement_per_sample_train.npy")

    ##Get the clean indexes in the last epoch
    clean_idx = clean_idx[-1, :]


    train_loader, test_loader, clean_labels, noisy_labels, noisy_indexes, trainset, all_labels = data_config(args, transform_train, transform_test, clean_idx)
    st = time.time()


    model = mod.PreActResNet18(num_classes=num_classes, low_dim=args.low_dim, head=args.headType, mode=args.method).to(device)

    model.load_state_dict(torch.load(
        "noise_models_" + args.network + "_Test_MOIT_" + args.noise_type + "_SI" + str(
            args.seed_initialization) + "_" + "SD" + str(args.seed_dataset) +
        "/" + str(args.noise_ratio) + "/MOIT_model.pth"))


    if args.ReInitializeClassif==1:
        model.linear2 = nn.Linear(512, num_classes).to(device)


    print('Total params: {:.2f} M'.format((sum(p.numel() for p in model.parameters()) / 1000000.0)))

    milestones = args.M

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


    loss_train_epoch = []
    loss_val_epoch = []

    acc_train_per_epoch = []
    acc_val_per_epoch = []
    loss_per_epoch_train = []


    exp_path = os.path.join('./', 'noise_models_' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name,
                                                                                             args.seed_initialization,
                                                                                             args.seed_dataset),
                            str(args.noise_ratio))
    res_path = os.path.join('./', 'metrics' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name,
                                                                                       args.seed_initialization,
                                                                                       args.seed_dataset),
                            str(args.noise_ratio))

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    np.save(res_path + '/' + str(args.noise_ratio) + '_true_labels.npy', np.asarray(clean_labels))
    np.save(res_path + '/' + str(args.noise_ratio) + '_noisy_labels.npy', np.asarray(noisy_labels))
    np.save(res_path + '/' + str(args.noise_ratio) + '_diff_labels.npy', noisy_indexes)
    np.save(res_path + '/' + str(args.noise_ratio) + '_all_labels.npy', all_labels)


    cont = 0


    for epoch in range(args.initial_epoch, args.epoch + 1):
        print("=================>    ", args.experiment_name, args.noise_ratio)


        scheduler.step()


        loss_per_epoch, top_5_train_ac, top1_train_ac, train_time = train_mixupBoot(args, model, device, train_loader, optimizer, epoch)

        loss_train_epoch += [loss_per_epoch]


        loss_per_epoch_val, acc_val_epoch_i = test_eval(args, model, device, test_loader)

        loss_val_epoch += [loss_per_epoch_val]
        acc_train_per_epoch += [top1_train_ac]
        acc_val_per_epoch += [acc_val_epoch_i]


        print('Epoch time: {:.2f} seconds\n'.format(time.time()-st))
        st = time.time()

        if epoch == args.initial_epoch:
            best_acc_val = acc_val_epoch_i
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestAccVal_%.5f' % (
                epoch, loss_per_epoch_val, acc_val_epoch_i, args.noise_ratio, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
        else:
            if acc_val_per_epoch[-1] > best_acc_val:
                best_acc_val = acc_val_per_epoch[-1]

                if cont > 0:
                    try:
                        os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                        os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    except OSError:
                        pass
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestAccVal_%.5f' % (
                    epoch, loss_val_epoch[-1], acc_val_per_epoch[-1], args.noise_ratio, best_acc_val)
                torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
                torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))


        cont += 1

        if epoch == args.epoch:
            snapLast = args.method + "_model"
            torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))



        # Save losses:
        np.save(res_path + '/' + str(args.noise_ratio) + '_LOSS_epoch_train.npy', np.asarray(loss_train_epoch))
        np.save(res_path + '/' + str(args.noise_ratio) + '_LOSS_epoch_val.npy', np.asarray(loss_val_epoch))

        # save accuracies:
        np.save(res_path + '/' + str(args.noise_ratio) + '_accuracy_per_epoch_train.npy',
                np.asarray(acc_train_per_epoch))
        np.save(res_path + '/' + str(args.noise_ratio) + '_accuracy_per_epoch_val.npy', np.asarray(acc_val_per_epoch))

        # save individual losses per epoch
        np.save(res_path + '/' + str(args.noise_ratio) + '_losses_per_epoch_train.npy', np.asarray(loss_per_epoch_train))


    print('Best ac:%f' % best_acc_val)


if __name__ == "__main__":
    args = parse_args()
    logging.info(args)
    # record params

    # train
    main(args)
