from os import listdir, makedirs
from os.path import exists, join
import argparse

import cv2
from skimage.filters import threshold_otsu
from skimage.io import imread
from tqdm import tqdm
from PIL.Image import fromarray, Image
from PIL.Image import Image as im
from PIL import Image
from numpy import asarray, repeat

import utils
from utils import extract_patches_2d
import numpy as np

import warnings

warnings.filterwarnings("ignore")

startswith = str.startswith
split = str.split
save = im.save
array = np.array


########################################################################################################################
#                                                   ICDAR2017                                                          #
########################################################################################################################

def create_patches_dataset_icdar17(data_path, new_path, height=256, width=256, num_patches=100, split=[3, 1, 1],
                                   seed=None, binary=False, stride=1):
    file_names = listdir(data_path)
    split_names = ['train', 'validation', 'test']
    dir_split = list(repeat(split_names, split))

    if not exists(new_path):
        makedirs(new_path)

    all_labels = [file_name.split('-')[0] for file_name in file_names]
    labels = list(dict.fromkeys(all_labels))

    for label in tqdm(labels):
        img_set = [f for f in file_names if f.startswith(label + '-')]

        for num, img_name in enumerate(img_set):

            if binary:
                img = text_padding2(data_path + img_name, max_width=1455, max_height=2580)
                # otsu = threshold_otsu(img)
                # img = img.point(lambda x: 255 if x < otsu else 0, '1')
            else:
                img = text_padding2(data_path + img_name, max_width=1455, max_height=2580)

            img = asarray(img)
            label_dir = join(new_path, dir_split[num], label)

            if not exists(label_dir):
                makedirs(label_dir)

            if dir_split[num] == 'test':
                patches = extract_patches_2d(img, (height, width), random_state=seed, stride=(height, width))
            else:
                patches = extract_patches_2d(img, (height, width), num_patches, random_state=seed, stride=stride)

            i = 1
            for p in range(len(patches)):
                patch = patches[p]
                img = fromarray(patch)
                img_dir = join(label_dir, img_name.split('.')[0] + '_patch_' + str(i) + '.jpg')

                if binary:
                    img.save(img_dir, mode=1, optimize=True)
                else:
                    img.save(img_dir)

                i = i + 1


def create_pages_dataset_icdar17(data_path, new_path, split=[3, 1, 1]):
    file_names = listdir(data_path)
    split_names = ['train', 'validation', 'test']
    dir_split = list(repeat(split_names, split))

    if not exists(new_path):
        makedirs(new_path)

    all_labels = [file_name.split('-')[0] for file_name in file_names]
    labels = list(dict.fromkeys(all_labels))

    for label in tqdm(labels):
        img_set = [f for f in file_names if f.startswith(label + '-')]

        for num, img_name in enumerate(img_set):
            img = text_padding2(data_path + img_name, max_width=1455, max_height=2580, color=(255,) * 3)
            label_dir = join(new_path, dir_split[num], label)

            if not exists(label_dir):
                makedirs(label_dir)

            img_dir = join(label_dir, img_name.split('.')[0] + '.jpg')

            img.save(img_dir)


########################################################################################################################
#                                                   FIREMAKER                                                          #
########################################################################################################################

def crop_firemaker_train():
    # firemaker-train folder corresponds to firemaker page one folder
    path = '/home/akshay/PycharmProjects/TFG/datasets/firemaker-train/'
    new_path = '/home/akshay/PycharmProjects/TFG/datasets/crop-firemaker-train/'

    left = 50
    top = 700
    right = 2400
    bottom = 3250

    if not exists(new_path):
        makedirs(new_path)

    for file_name in listdir(path):
        img = Image.open(path + file_name)
        cropped_img = img.crop((left, top, right, bottom))
        img_dir = join(new_path, file_name)
        cropped_img.save(img_dir)


def crop_firemaker_test():
    # firemaker-test folder corresponds to firemaker page four folder
    path = '/home/akshay/PycharmProjects/TFG/datasets/firemaker-test/'
    new_path = '/home/akshay/PycharmProjects/TFG/datasets/crop-firemaker-test/'

    left = 50
    top = 700
    right = 2400
    bottom = 1975

    if not exists(new_path):
        makedirs(new_path)

    for file_name in listdir(path):
        img = Image.open(path + file_name)
        cropped_img = img.crop((left, top, right, bottom))
        img_dir = join(new_path, file_name)
        cropped_img.save(img_dir)


def text_padding3(img, max_width=2500, max_height=2100, color=(255,) * 3):
    height, width = img.shape

    height_pad = cv2.copyMakeBorder(img, (max_height - height) // 2, (max_height - height + 1) // 2, 0,
                                    (max_width - width),
                                    cv2.BORDER_WRAP)
    return fromarray(height_pad)


def text_padding_numpy(path, max_width=2500, max_height=2100, color=(255,) * 3):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    height, width, _ = img.shape

    height_pad = cv2.copyMakeBorder(img, (max_height - height) // 2, (max_height - height + 1) // 2, 0,
                                    (max_width - width),
                                    cv2.BORDER_WRAP)
    return height_pad


def text_padding2(path, max_width=2500, max_height=2100, color=(255,) * 3):
    img = cv2.imread(path)
    height, width, _ = img.shape

    height_pad = cv2.copyMakeBorder(img, (max_height - height) // 2, (max_height - height + 1) // 2, 0,
                                    (max_width - width),
                                    cv2.BORDER_WRAP)
    return fromarray(height_pad)


def text_padding(path, max_width=2500, max_height=2100, color=(255,) * 3):
    img = cv2.imread(path)
    height, width, _ = img.shape

    height_pad = cv2.copyMakeBorder(img, (max_height - height) // 2, (max_height - height + 1) // 2, 0, 0,
                                    cv2.BORDER_WRAP)
    # Get text regions
    gray = cv2.cvtColor(height_pad, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (500, 100))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # copy regions at the end of lines
    width_pad = cv2.copyMakeBorder(height_pad, 0, 0, 0, (max_width - width), cv2.BORDER_CONSTANT, value=color)
    for c in cnts:
        if np.prod(c.shape) < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ROI = width_pad[y:y + h, x:x + w]
        pad = ROI[:, 0:width_pad.shape[1] - ROI.shape[1]].copy()
        width_pad[y:y + h, w:x + w + pad.shape[1]] = pad.copy()
    return fromarray(width_pad)


def create_pages_dataset_firemaker(train_path='/home/akshay/PycharmProjects/TFG/datasets/crop-firemaker-train/',
                                   test_path='/home/akshay/PycharmProjects/TFG/datasets/crop-firemaker-test/',
                                   new_path='/home/akshay/PycharmProjects/TFG/datasets/pages-firemaker/'):
    if not exists(new_path):
        makedirs(new_path)

    train_file_names = sorted(listdir(train_path))
    test_file_names = sorted(listdir(test_path))
    for i in tqdm(range(len(train_file_names))):
        train_label_dir = join(new_path, 'train', str(i))
        test_label_dir = join(new_path, 'test', str(i))

        if not exists(train_label_dir):
            makedirs(train_label_dir)

        if not exists(test_label_dir):
            makedirs(test_label_dir)

        train_img_dir = join(train_label_dir, train_file_names[i])
        test_img_dir = join(test_label_dir, test_file_names[i])

        train_img = text_padding(train_path + train_file_names[i], max_width=2500, max_height=2100)
        test_img = text_padding(test_path + test_file_names[i], max_width=2500, max_height=2100)

        train_img.save(train_img_dir)
        test_img.save(test_img_dir)


def create_patches_dataset_firemaker(train_path, test_path, new_path, height=256, width=256, num_patches=100,
                                     seed=None, binary=False, stride=1):
    if not exists(new_path):
        makedirs(new_path)

    train_file_names = sorted(listdir(train_path))
    test_file_names = sorted(listdir(test_path))
    for t in tqdm(range(len(train_file_names))):

        train_label_dir = join(new_path, 'train', str(t))
        validation_label_dir = join(new_path, 'validation', str(t))
        test_label_dir = join(new_path, 'test', str(t))

        if not exists(train_label_dir):
            makedirs(train_label_dir)

        if not exists(validation_label_dir):
            makedirs(validation_label_dir)

        if not exists(test_label_dir):
            makedirs(test_label_dir)

        if binary:
            train_img = text_padding_numpy(train_path + train_file_names[t])
            otsu = threshold_otsu(train_img)
            train_img = train_img < otsu
            test_img = text_padding_numpy(test_path + test_file_names[t])
            otsu = threshold_otsu(test_img)
            test_img = test_img < otsu
        else:
            train_img = text_padding2(train_path + train_file_names[t], max_width=2480, max_height=2100)
            test_img = text_padding2(test_path + test_file_names[t], max_width=2480, max_height=2100)

        train_img = asarray(train_img)
        test_img = asarray(test_img)

        train_and_val_patches = extract_patches_2d(train_img, (height, width), num_patches, random_state=seed,
                                                   stride=stride)
        train_patches, val_patches = utils.train_val_split(train_and_val_patches, train_ratio=0.9)
        test_patches = extract_patches_2d(test_img, (height, width), random_state=seed, stride=(height, width))

        a = 1
        for p in range(len(train_patches)):
            patch = train_patches[p]
            img = fromarray(patch)
            img_dir = join(train_label_dir, train_file_names[t].split('.')[0] + '_patch_' + str(a) + '.jpg')

            if binary:
                img.save(img_dir, mode=1, optimize=True)
            else:
                img.save(img_dir)
            a = a + 1

        b = 1
        for p in range(len(val_patches)):
            patch = val_patches[p]
            img = fromarray(patch)
            img_dir = join(validation_label_dir, train_file_names[t].split('.')[0] + '_patch_' + str(b) + '.jpg')

            if binary:
                img.save(img_dir, mode=1, optimize=True)
            else:
                img.save(img_dir)
            b = b + 1

        c = 1
        for p in range(len(test_patches)):
            patch = test_patches[p]
            img = fromarray(patch)
            img_dir = join(test_label_dir, test_file_names[t].split('.')[0] + '_patch_' + str(c) + '.jpg')

            if binary:
                img.save(img_dir, mode=1, optimize=True)
            else:
                img.save(img_dir)
            c = c + 1


########################################################################################################################
#                                                   IAM                                                                #
########################################################################################################################

def crop_iam(path='/home/akshay/PycharmProjects/TFG/datasets/mini-IAM/',
             new_path='/home/akshay/PycharmProjects/TFG/datasets/crop-mini-IAM/'):
    left = 50
    top = 700
    right = 2400
    bottom = 2700

    if not exists(new_path):
        makedirs(new_path)

    for file_name in tqdm(listdir(path)):
        img = Image.open(path + file_name)
        cropped_img = img.crop((left, top, right, bottom))
        img_dir = join(new_path, file_name)
        cropped_img.save(img_dir)


def get_labels_iam(path='/home/punjabi/TFG/datasets/forms.txt'):
    d = {}
    with open(path) as f:
        for line in f:
            l = []
            img = line.split(' ')[0] + '.png'
            writer = int(line.split(' ')[1])
            l = d.get(writer, l)
            if len(l) < 2:
                l.append(img)
                d[writer] = l
    return d


def create_pages_dataset_iam(train_path='/home/punjabi/TFG/datasets/crop-IAM/',
                             new_path='/home/punjabi/TFG/datasets/pages-IAM/'):
    if not exists(new_path):
        makedirs(new_path)

    d = get_labels_iam()
    for label, imgs in tqdm(d.items()):
        train_label_dir = join(new_path, 'train', str(label))
        test_label_dir = join(new_path, 'test', str(label))

        if not exists(train_label_dir):
            makedirs(train_label_dir)

        if not exists(test_label_dir):
            makedirs(test_label_dir)

        train_img_dir = join(train_label_dir, imgs[0])
        try:
            test_img_dir = join(test_label_dir, imgs[1])
        except IndexError:
            test_img_dir = join(test_label_dir, imgs[0])

        if len(imgs) == 1:
            img = Image.open(train_path + imgs[0])
            img = asarray(img)
            train_img, test_img = halve_image(img)
            train_img = text_padding3(train_img, max_width=2350, max_height=2195)
            test_img = text_padding3(test_img, max_width=2350, max_height=2195)
        else:
            train_img = text_padding2(train_path + imgs[0], max_width=2350, max_height=2195)
            test_img = text_padding2(train_path + imgs[1], max_width=2350, max_height=2195)

        train_img.save(train_img_dir)
        test_img.save(test_img_dir)


def create_patches_dataset_iam(data_path, new_path, height=100, width=100, num_patches=10,
                               seed=None, binary=False, stride=1):
    if not exists(new_path):
        makedirs(new_path)

    d = get_labels_iam()
    for label, imgs in tqdm(d.items()):
        train_label_dir = join(new_path, 'train', str(label))
        validation_label_dir = join(new_path, 'validation', str(label))
        test_label_dir = join(new_path, 'test', str(label))

        if not exists(train_label_dir):
            makedirs(train_label_dir)

        if not exists(validation_label_dir):
            makedirs(validation_label_dir)

        if not exists(test_label_dir):
            makedirs(test_label_dir)

        if len(imgs) == 1:
            if binary:
                img = imread(data_path + imgs[0], as_gray=True, plugin='pil')
                otsu = threshold_otsu(img)
                img = img < otsu
            else:
                img = Image.open(data_path + imgs[0])

            img = asarray(img)
            train_img, test_img = halve_image(img)
            train_img = text_padding3(train_img, max_width=2350, max_height=2195)
            test_img = text_padding3(test_img, max_width=2350, max_height=2195)

        else:
            if binary:
                train_img = text_padding_numpy(data_path + imgs[0])
                otsu = threshold_otsu(train_img)
                train_img = train_img < otsu
                test_img = text_padding_numpy(data_path + imgs[1])
                otsu = threshold_otsu(test_img)
                test_img = test_img < otsu
            else:
                train_img = text_padding2(data_path + imgs[0], max_width=2350, max_height=2195)
                test_img = text_padding2(data_path + imgs[1], max_width=2350, max_height=2195)

        train_img = asarray(train_img)
        test_img = asarray(test_img)

        train_and_val_patches = extract_patches_2d(train_img, (height, width), num_patches, random_state=seed,
                                                   stride=stride, th=1000)
        train_patches, val_patches = utils.train_val_split(train_and_val_patches, train_ratio=0.9)
        test_patches = extract_patches_2d(test_img, (height, width), random_state=seed, stride=(height, width), th=1000)

        a = 1
        for p in range(len(train_patches)):
            patch = train_patches[p]
            img = fromarray(patch)
            img_dir = join(train_label_dir, imgs[0].split('.')[0] + '_patch_' + str(a) + '.jpg')

            if binary:
                img.save(img_dir, mode=1, optimize=True)
            else:
                img.save(img_dir)
            a = a + 1

        b = 1
        for p in range(len(val_patches)):
            patch = val_patches[p]
            img = fromarray(patch)
            img_dir = join(validation_label_dir, imgs[0].split('.')[0] + '_patch_' + str(b) + '.jpg')

            if binary:
                img.save(img_dir, mode=1, optimize=True)
            else:
                img.save(img_dir)
            b = b + 1

        c = 1
        for p in range(len(test_patches)):
            patch = test_patches[p]
            img = fromarray(patch)
            try:
                img_dir = join(test_label_dir, imgs[1].split('.')[0] + '_patch_' + str(c) + '.jpg')
            except IndexError:
                img_dir = join(test_label_dir, imgs[0].split('.')[0] + '_patch_' + str(c) + '.jpg')

            if binary:
                img.save(img_dir, mode=1, optimize=True)
            else:
                img.save(img_dir)
            c = c + 1


def halve_image(img):
    height, width = img.shape[:2]
    left, top, bottom, right = int(0), int(0), int(height * .5), int(width)
    cropped_img_top = img[left:bottom, top:right]

    left, top, bottom, right = int(height * .5), int(0), int(height), int(width)
    cropped_img_bottom = img[left:bottom, top:right]
    return cropped_img_top, cropped_img_bottom


########################################################################################################################
#                                                   MAIN                                                               #
########################################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data-path', metavar='DIR', type=str,
                        default='/home/punjabi/TFG/datasets/ScriptNet-HistoricalWI-2017-color/',
                        help='path to dataset')
    parser.add_argument('--test-path', metavar='DIR', type=str,
                        help='path to dataset')
    parser.add_argument('--new-path', metavar='DIR', type=str,
                        default='/home/punjabi/TFG/datasets/pages-ScriptNet-HistoricalWI-2017-color/',
                        help='path to new dataset')
    parser.add_argument('--patch-height', default=256, type=int, metavar='H',
                        help='height of the patch')
    parser.add_argument('--patch-width', default=256, type=int, metavar='W',
                        help='width of the patch')
    parser.add_argument('--num-patches', default=500, type=int, metavar='N',
                        help='number of patches per image')
    parser.add_argument('--split', default=[3, 1, 1], type=int, nargs='+', metavar='S',
                        help='number of images for train (first element), val (second element) and test (third)')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for choosing random patches. ')
    parser.add_argument('--binary', dest='binary', action='store_true',
                        help='create images in binary, experiments show is better')
    parser.add_argument('--stride', default=1,
                        help='factor to make the binary difference. ')

    parser.add_argument('--pages', dest='pages', action='store_true',
                        help='save images as pages, not use patches')
    parser.add_argument('--dataset', type=str, default='icdar17', choices=('icdar17', 'firemaker', 'iam'),
                        help='supports three  icdar17, firemaker, iam')
    args = parser.parse_args()

    print('Creating dataset...')
    if args.dataset == 'icdar17':
        if args.pages:
            create_pages_dataset_icdar17(args.data_path, args.new_path, args.split)
        else:
            create_patches_dataset_icdar17(args.data_path, args.new_path, args.patch_height, args.patch_width,
                                           args.num_patches,
                                           seed=args.seed, binary=args.binary, stride=args.stride)
    elif args.dataset == 'firemaker':
        if args.pages:
            create_pages_dataset_firemaker(args.data_path, args.test_path, args.new_path)
        else:
            create_patches_dataset_firemaker(args.data_path, args.test_path, args.new_path, args.patch_height,
                                             args.patch_width, args.num_patches,
                                             seed=args.seed, binary=args.binary, stride=args.stride)
    else:
        if args.pages:
            create_pages_dataset_iam(args.data_path, args.new_path)
        else:
            create_patches_dataset_iam(args.data_path, args.new_path, args.patch_height, args.patch_width,
                                       args.num_patches,
                                       args.seed, args.binary, args.stride)
    print('Finished.')
