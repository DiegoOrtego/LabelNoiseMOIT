import os
import pickle
import torchvision as tv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import time
from IPython import embed

def get_dataset(args, transform_train, transform_test):
    # prepare datasets

    if args.validation_exp == 1:
        #################################### Train set #############################################
        temp_dataset = Cifar100Train(args, train=True, transform=transform_train, target_transform=transform_test, download=args.download)
        train_indexes, val_indexes = train_val_split(args, temp_dataset.targets)
        cifar_train = Cifar100Train(args, train=True, transform=transform_train, target_transform=transform_test, sample_indexes=train_indexes)
        #################################### Noise corruption ######################################
        if args.noise_type == 'symmetric':
            cifar_train.random_in_noise()

        elif args.noise_type == 'asymmetric':
            cifar_train.real_in_noise(train_indexes)

        else:
            print('No noise')

        cifar_train.labelsNoisyOriginal = cifar_train.targets.copy()
        #################################### Test set #############################################
        testset = Cifar100Train(args, train=True, transform=transform_test, sample_indexes=val_indexes)
        ###########################################################################################

    else:
        #################################### Train set #############################################
        cifar_train = Cifar100Train(args, train=True, transform=transform_train, target_transform=transform_test, download=args.download)
        #################################### Noise corruption ######################################
        if args.noise_type == 'symmetric':
            cifar_train.random_in_noise()

        elif args.noise_type == 'asymmetric':
            cifar_train.real_in_noise([])

        else:
            print('No noise')

        cifar_train.labelsNoisyOriginal = cifar_train.targets.copy()

        #################################### Test set #############################################
        testset = tv.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        ###########################################################################################

    return cifar_train, testset, cifar_train.clean_labels, cifar_train.noisy_labels, cifar_train.noisy_indexes,  cifar_train.labelsNoisyOriginal

def train_val_split(args, train_val):
    np.random.seed(args.seed_dataset)
    train_val = np.array(train_val)
    train_indexes = []
    val_indexes = []
    val_num = int(args.val_samples / args.num_classes)

    for id in range(args.num_classes):
        indexes = np.where(train_val == id)[0]
        np.random.shuffle(indexes)
        val_indexes.extend(indexes[:val_num])
        train_indexes.extend(indexes[val_num:])
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)

    return train_indexes, val_indexes

class Cifar100Train(tv.datasets.CIFAR100):
    def __init__(self, args, train=True, transform=None, target_transform=None, sample_indexes=None, download=False):
        super(Cifar100Train, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        self.args = args
        if sample_indexes is not None:
            self.data = self.data[sample_indexes]
            self.targets = list(np.asarray(self.targets)[sample_indexes])

        self.num_classes = self.args.num_classes
        self.in_index = []
        self.out_index = []
        self.noisy_indexes = []
        self.clean_indexes = []
        self.clean_labels = []
        self.noisy_labels = []
        self.out_data = []
        self.out_labels = []
        self.soft_labels = []
        self.labelsNoisyOriginal = []
        self._num = []
        self._count = 1
        self.prediction = []
        self.confusion_matrix_in = np.array([])
        self.confusion_matrix_out = np.array([])
        self.labeled_idx = []
        self.unlabeled_idx = []


        # From in ou split function:
        self.soft_labels = np.zeros((len(self.targets), self.num_classes), dtype=np.float32)
        self._num = int(len(self.targets) * self.args.noise_ratio)


    ################# Symmetric noise #########################
    def random_in_noise(self):

        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)
        idxes = np.random.permutation(len(self.targets))
        clean_labels = np.copy(self.targets)
        noisy_indexes = idxes[0:self._num]
        clean_indexes = idxes[self._num:]
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 0 ## Remove soft-label created during label mapping
                # train_labels[idxes[i]] -> another category
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                while(label_sym==self.targets[idxes[i]]):#To exclude the original label
                    label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.targets[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.targets[idxes[i]]] = 1

        self.targets = np.asarray(self.targets, dtype=np.long)
        self.noisy_labels = np.copy(self.targets)
        self.noisy_indexes = noisy_indexes
        self.clean_labels = clean_labels
        self.clean_indexes = clean_indexes
        self.confusion_matrix_in = (np.ones((self.args.num_classes, self.args.num_classes)) - np.identity(self.args.num_classes))\
                                    *(self.args.noise_ratio/(self.num_classes -1)) + \
                                    np.identity(self.args.num_classes)*(1 - self.args.noise_ratio)


    ##########################################################################


    ################# Asymmetric noise #########################

    def real_in_noise(self, train_indexes):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)

        ##### Create te confusion matrix #####

        self.confusion_matrix_in = np.identity(self.args.num_classes) * (1 - self.args.noise_ratio)

        idxes = np.random.permutation(len(self.targets))
        clean_labels = np.copy(self.targets)

        with open('data/cifar-100-python/train', 'rb') as f:
            entry = pickle.load(f, encoding='latin1')

        coarse_targets = np.asarray(entry['coarse_labels'])

        if self.args.validation_exp==1:
            coarse_targets = coarse_targets[train_indexes]


        targets = np.array(self.targets)
        num_subclasses = self.args.num_classes // 20

        for i in range(20):
            # embed()
            subclass_targets = np.unique(targets[coarse_targets == i])
            clean = subclass_targets
            noisy = np.concatenate([clean[1:], clean[:1]])
            for j in range(num_subclasses):
                self.confusion_matrix_in[clean[j], noisy[j]] = self.args.noise_ratio



        for t in range(len(idxes)):
            self.soft_labels[idxes[t]][self.targets[idxes[t]]] = 0  ## Remove soft-label created during label mapping
            current_label = self.targets[idxes[t]]
            conf_vec = self.confusion_matrix_in[current_label,:]
            label_sym = np.random.choice(np.arange(0, self.num_classes), p=conf_vec.transpose())
            self.targets[idxes[t]] = label_sym
            self.soft_labels[idxes[t]][self.targets[idxes[t]]] = 1

            if label_sym == current_label:
                self.clean_indexes.append(idxes[t])
            else:
                self.noisy_indexes.append(idxes[t])

        self.targets = np.asarray(self.targets, dtype=np.long)
        self.clean_indexes = np.asarray(self.clean_indexes, dtype=np.long)
        self.noisy_indexes = np.asarray(self.noisy_indexes, dtype=np.long)
        self.noisy_labels = self.targets
        self.clean_labels = clean_labels

    def __getitem__(self, index):
        if len(self.data) > self.args.val_samples:
            img, labels, soft_labels, noisy_labels, clean_labels = self.data[index], self.targets[index], \
                                                                   self.soft_labels[
                                                                       index], self.labelsNoisyOriginal[index], \
                                                                   self.clean_labels[index]

            img = Image.fromarray(img)

            img_noDA = self.target_transform(img)

            img1 = self.transform(img)

            if self.args.method == "MOIT":
                if self.train is True:
                    img2 = self.transform(img)
                else:
                    img2 = 0

                return img1, img2, img_noDA, labels, soft_labels, index, noisy_labels, clean_labels

            else:
                return img1, img_noDA, labels, soft_labels, index, noisy_labels, clean_labels

        else:
            img, labels = self.data[index], self.targets[index]

            img = Image.fromarray(img)

            img = self.transform(img)

            return img, labels

