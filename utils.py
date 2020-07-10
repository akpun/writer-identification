import numbers

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from numpy import asarray
import numpy as np
from numpy.lib.stride_tricks import as_strided
from skimage import feature
from skimage.filters import threshold_otsu
from sklearn.utils import check_random_state, check_array
from torch import sqrt
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_val_split(data, train_ratio=0.9):
    train_size = int(train_ratio * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    return train_data, val_data


def binary(img):
    gray_img = img.convert('L')
    otsu = threshold_otsu(asarray(gray_img))
    binary_img = gray_img.point(lambda x: 255 if x < otsu else 0, '1')
    return binary_img


class Binary(object):
    def __call__(self, img):
        return binary(img)


def squeeze_weights(m):
    m.weight.data = m.weight.data.sum(dim=1)[:, None]
    m.in_channels = 1


def change_out_features(m, classes):
    m.out_features = classes
    return m


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def dataset_mean_and_std(train_path, test_path):
    # Dataset should be a folder which follows
    # ImageFolder format with pages in each label folder

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = ImageFolder(train_path,
                             transform=transform)
    test_data = ImageFolder(test_path,
                            transform=transform)
    data = ConcatDataset([train_data, test_data])
    loader = DataLoader(data, batch_size=1)

    n = 0
    m = 0.0
    var = 0.0
    with tqdm(total=len(loader)) as pbar:
        for data in loader:
            batch = data[0]
            # Rearrange batch to be the shape of [B, C, W * H]
            batch = batch.view(batch.size(0), batch.size(1), -1)
            # Update total number of images
            n += batch.size(0)
            # Compute mean and std here
            m += batch.mean(2).sum(0)
            var += batch.var(2).sum(0)
            pbar.update(1)

    m /= n
    var /= n
    s = sqrt(var)

    print(m)
    print(s)

    return m, s


def _extract_patches(arr, patch_shape=8, extraction_step=1):
    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches


def extract_patches_2d(image, patch_size, max_patches=None,
                       random_state=None, stride=1, th=2000):
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    if isinstance(stride, numbers.Number):
        step = stride
        s_h = stride
        s_w = stride
    else:
        s_h, s_w = stride
        step = (s_h, s_w, n_colors)

    extracted_patches = _extract_patches(image,
                                         patch_shape=(p_h, p_w, n_colors),
                                         extraction_step=step)

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, stride, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint((i_h - p_h + 1) // s_h, size=n_patches)
        j_s = rng.randint((i_w - p_w + 1) // s_w, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        patches = patches.reshape((n_patches, p_h, p_w))

    # return clean_patches(patches, th)
    return patches


def _compute_n_patches(i_h, i_w, p_h, p_w, stride, max_patches=None):
    if isinstance(stride, numbers.Number):
        s_h = stride
        s_w = stride
    else:
        s_h, s_w = stride

    n_h = (i_h - p_h) // s_h + 1
    n_w = (i_w - p_w) // s_w + 1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, numbers.Integral)
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, numbers.Integral)
              and max_patches >= all_patches):
            return all_patches
        elif (isinstance(max_patches, numbers.Real)
              and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def clean_patches(patches, th=2000):
    indices = []
    for i, patch in enumerate(patches):
        if patch.shape[-1] == 3:
            patch = patch / 255
            num_features = feature.canny(patch.mean(axis=2), sigma=2).sum()
        else:
            num_features = feature.canny(patch, sigma=2).sum()

        if num_features > th:
            indices.append(i)
    return patches[indices]


def get_labels_and_class_counts(labels_list):
    '''
    Calculates the counts of all unique classes.
    '''
    labels = np.array(labels_list)
    _, class_counts = np.unique(labels, return_counts=True)
    return labels, class_counts


def plot_class_distributions(class_names, train_class_counts,
                             test_class_counts, validation_class_counts):
    '''
    Plots the class distributions for the training and test set asa barplot.
    '''
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True, figsize=(15, 6))
    ax1.bar(class_names, train_class_counts)
    ax1.set_title('Training dataset distribution')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Class counts')
    ax2.bar(class_names, test_class_counts)
    ax2.set_title('Test dataset distribution')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Class counts')
    ax3.bar(class_names, validation_class_counts)
    ax3.set_title('Validation dataset distribution')
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Class counts')


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        elif self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, BinColorDataset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class BinColorDataset(Dataset):
    def __init__(self, dataset, col_transform=None, bin_transform=None):
        self.dataset = dataset
        self.col_transform = col_transform
        self.bin_transform = bin_transform

    def __getitem__(self, index):
        x1, y1 = self.dataset[index]

        if self.bin_transform:
            x2 = self.bin_transform(x1)
        if self.col_transform:
            x1 = self.col_transform(x1)

        return x1, x2, y1

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.7993, 0.7404, 0.6438], [0.1168, 0.1198, 0.1186]),  # icdar17 norm
    # ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.9706, 0.9706, 0.9706], [0.1448, 0.1448, 0.1448]),  # firemaker norm
    ])

    train_path = '/home/akshay/PycharmProjects/TFG/datasets/firemaker-500/train'
    val_path = '/home/akshay/PycharmProjects/TFG/datasets/firemaker-500/validation'
    test_path = '/home/akshay/PycharmProjects/TFG/datasets/firemaker-500/test'

    train_data = ImageFolder(train_path, transform=transform)

    val_data = ImageFolder(val_path, transform=transform)

    test_data = ImageFolder(test_path, transform=transform)
    labels, c1 = get_labels_and_class_counts(train_data.targets)
    labels1, c2 = get_labels_and_class_counts(test_data.targets)
    labels2, c3 = get_labels_and_class_counts(val_data.targets)
    plot_class_distributions(train_data.classes, c1, c2, c3)
