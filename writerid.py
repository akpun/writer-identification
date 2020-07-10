import argparse
import os
from os.path import join
from tempfile import TemporaryDirectory
from collections import OrderedDict
import numpy as np

import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import utils
from create_datasets import create_patches_dataset_icdar17, create_pages_dataset_icdar17, \
    create_pages_dataset_firemaker, create_patches_dataset_firemaker, create_patches_dataset_iam

MODEL_NAMES = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


class WriterID(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.dataset = hparams.dataset

        if self.dataset == 'icdar17':
            self.num_classes = 720
        elif self.dataset == 'firemaker':
            self.num_classes = 250
        else:
            self.num_classes = 657

        self.model = models.__dict__[self.hparams.model](pretrained=hparams.pretrained)
        # self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        # Changing first layer
        if hparams.binary:
            name, layer = list(self.model.named_children())[0]
            if type(layer) == torch.nn.modules.container.Sequential:
                layer[0].apply(utils.squeeze_weights)
                setattr(self.model, name, layer)
            else:
                setattr(self.model, name, layer.apply(utils.squeeze_weights))

        # Changing last layer
        name, layer = list(self.model.named_children())[-1]
        if type(layer) == torch.nn.modules.container.Sequential:
            utils.change_out_features(layer[-1], classes=self.num_classes)
            setattr(self.model, name, layer)
        else:
            setattr(self.model, name, utils.change_out_features(layer, classes=self.num_classes))

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        train_loss = F.cross_entropy(y_pred, y)
        acc1, acc5, acc10 = self.__accuracy(y_pred, y, topk=(1, 5, 10))

        logs = {'train_loss': train_loss}
        output = OrderedDict({
            'loss': train_loss,
            'acc1': acc1,
            'acc5': acc5,
            'acc10': acc10,
            'log': logs
        })

        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = F.cross_entropy(y_pred, y)
        y_pred = F.softmax(y_pred, dim=-1)

        acc1, acc5, acc10 = self.__accuracy(y_pred, y, topk=(1, 5, 10))

        output = OrderedDict({
            'val_loss': val_loss,
            'val_acc1': acc1,
            'val_acc5': acc5,
            'val_acc10': acc10,
            'y_pred': y_pred,
            'y': y,
        })

        return output

    def validation_epoch_end(self, outputs):
        logs = {}

        for metric_name in ["val_loss", "val_acc1", "val_acc5", "val_acc10"]:
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]

                # reduce manually when using dp
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)

                metric_total += metric_value

            logs[metric_name] = metric_total / len(outputs)

        preds = []
        targets = []
        for output in outputs:
            y_pred = output['y_pred']
            y = output['y']

            preds.append(y_pred.cpu())
            targets.append(y.cpu())

        targets = torch.cat(targets, dim=0)
        preds = torch.cat(preds, dim=0)

        class_acc = self.__conf_matrix(preds, targets, self.classes)
        logs['class_acc'] = 100 * class_acc

        result = {'log': logs, 'val_loss': logs["val_loss"]}
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        test_loss = F.cross_entropy(y_pred, y)
        y_pred = F.softmax(y_pred, dim=-1)
        acc1, acc5, acc10 = self.__accuracy(y_pred, y, topk=(1, 5, 10))

        output = OrderedDict({
            'test_loss': test_loss,
            'test_acc1': acc1,
            'test_acc5': acc5,
            'test_acc10': acc10,
            'y_pred': y_pred,
            'y': y,
        })

        return output

    def test_epoch_end(self, outputs):
        logs = {}

        for metric_name in ["test_loss", "test_acc1", "test_acc5", "test_acc10"]:
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]

                # reduce manually when using dp
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)

                metric_total += metric_value

            logs[metric_name] = metric_total / len(outputs)

        preds = []
        targets = []
        for output in outputs:
            y_pred = output['y_pred']
            y = output['y']

            preds.append(y_pred.cpu())
            targets.append(y.cpu())

        targets = torch.cat(targets, dim=0)
        preds = torch.cat(preds, dim=0)

        class_acc = self.__conf_matrix(preds, targets, self.classes)
        logs['test_class_acc'] = 100 * class_acc

        result = {'log': logs, 'test_loss': logs["test_loss"]}
        return result

    @classmethod
    def __conf_matrix(cls, preds, targets, classes, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)

            _, pred = preds.topk(maxk, 1, True, True)

            conf_matrix = confusion_matrix(targets, pred)
            print(len(conf_matrix))
            max_pred = np.argmax(conf_matrix, axis=1)
            print(len(classes))
            class_acc = accuracy_score(np.array(classes)[max_pred], np.array(classes))

            return class_acc

    @classmethod
    def __accuracy(cls, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)

    #        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
    #        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #        return [optimizer], [scheduler]

    def prepare_data(self):

        if self.hparams.binary:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])
        else:
            if self.dataset == 'icdar17':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.7993, 0.7404, 0.6438], [0.1168, 0.1198, 0.1186]),  # icdar17 norm
                ])
            elif self.dataset == 'firemaker':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.9706, 0.9706, 0.9706], [0.1448, 0.1448, 0.1448]),  # firemaker norm
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.9496, 0.9406, 0.9406], [0.1076, 0.1076, 0.1076]),  # iam norm
                ])

        self.train_data = ImageFolder(self.hparams.train_path, transform=transform)
        self.classes = self.train_data.classes
        self.train_sampler = utils.ImbalancedDatasetSampler(self.train_data)

        self.val_data = ImageFolder(self.hparams.val_path, transform=transform)
        self.test_data = ImageFolder(self.hparams.test_path, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, sampler=self.train_sampler,
                          num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, num_workers=8, pin_memory=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('-a', '--model', metavar='MODEL', default='resnet18', choices=MODEL_NAMES,
                            help='model architecture: ' +
                                 ' | '.join(MODEL_NAMES) +
                                 ' (default: )')
        parser.add_argument('--epochs', default=20, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=None,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch-size', default=32, type=int,
                            metavar='N',
                            help='mini-batch size (default: 32), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model in imagenet')
        return parser


def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)

    parent_parser.add_argument('--patch-height', default=256, type=int, metavar='H',
                               help='height of the patch')
    parent_parser.add_argument('--patch-width', default=256, type=int, metavar='W',
                               help='width of the patch')
    parent_parser.add_argument('--num-patches', default=None, type=int, metavar='N',
                               help='number of patches per image')
    parent_parser.add_argument('--stride', default=1, type=int, metavar='B',
                               help='factor in which pixels are binarized')
    parent_parser.add_argument('--split', default=[3, 1, 1], type=int, nargs='+', metavar='S',
                               help='number of images for train (first element), val (second element) and test (third)')
    parent_parser.add_argument('--binary', dest='binary', action='store_true',
                               help='use binary photos')
    parent_parser.add_argument('--data-path', metavar='T', type=str,
                               default='/home/punjabi/TFG/datasets/ScriptNet-HistoricalWI-2017-binarized/',
                               help='path to dataset')
    parent_parser.add_argument('--dataset', metavar='D', type=str,
                               default='icdar17',
                               choices=('icdar17', 'firemaker', 'iam'),
                               help='three datasets: icdar17, firemaker, iam')
    parent_parser.add_argument('--train-path', metavar='T', type=str,
                               default='/home/punjabi/TFG/datasets/patches-ScriptNet-HistoricalWI-2017-binarized/train',
                               help='path to train dataset')
    parent_parser.add_argument('--val-path', metavar='V', type=str,
                               default='/home/punjabi/TFG/datasets/patches-ScriptNet-HistoricalWI-2017-binarized/validation',
                               help='path to validation dataset')
    parent_parser.add_argument('--test-path', metavar='TST', type=str,
                               default='/home/punjabi/TFG/datasets/patches-ScriptNet-HistoricalWI-2017-binarized/test',
                               help='path to test dataset')
    parent_parser.add_argument('--checkpoint-path', metavar='C', type=str,
                               default=None,
                               help='path to test dataset')
    parent_parser.add_argument('--temp-path', metavar='TMP', default=os.getcwd(), type=str,
                               help='path to save data temporary')
    parent_parser.add_argument('--use-temp-dir', dest='use_temp_dir', action='store_true',
                               help='create data on a temporary directory')
    parent_parser.add_argument('--gpus', type=int, default=2,
                               help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16-bit', dest='use_16_bit', action='store_true',
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('-t', '--test', dest='test', action='store_true',
                               help='evaluate model on test  set')
    parent_parser.add_argument('--load-checkpoint', dest='load', action='store_true',
                               help='load model and evaluate on test  set')

    parser = WriterID.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    if hparams.use_temp_dir:
        tmp_dir = TemporaryDirectory(dir=hparams.temp_path)
        if hparams.dataset == 'icdar17':
            create_patches_dataset_icdar17(hparams.data_path, tmp_dir.name, hparams.patch_height, hparams.patch_height,
                                           hparams.num_patches,
                                           seed=hparams.seed, binary=hparams.binary, stride=hparams.stride)
        elif hparams.dataset == 'firemaker':
            create_patches_dataset_firemaker(hparams.data_path, hparams.test_path, tmp_dir.name, hparams.patch_height,
                                             hparams.patch_height, hparams.num_patches,
                                             seed=hparams.seed, binary=hparams.binary, stride=hparams.stride)
        else:
            create_patches_dataset_iam(hparams.data_path, tmp_dir.name, hparams.patch_height, hparams.patch_height,
                                       hparams.num_patches,
                                       hparams.seed, hparams.binary, hparams.stride)

        hparams.train_path = join(tmp_dir.name, 'train')
        hparams.val_path = join(tmp_dir.name, 'validation')
        hparams.test_path = join(tmp_dir.name, 'test')

    model = WriterID(hparams)

    # Loggers
    logger = WandbLogger(project='TFG', tags=[hparams.dataset])
    logger.watch(model)  # Watch gradients

    pl.seed_everything(hparams.seed)

    # Callbacks
    # early_stopping = EarlyStopping('val_loss', patience=30)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd() + '/checkpoints/' + hparams.dataset + '_' + hparams.model + '_{epoch:02d}-{val_loss:.2f}',
    )
    # checkpoint_callback = False

    trainer = pl.Trainer(
        nb_sanity_val_steps=0,
        train_path=hparams.train_path,
        val_path=hparams.val_path,
        test_path=hparams.test_path,
        gpus=hparams.gpus,
        batch_size=hparams.batch_size,
        learning_rate=hparams.lr,
        epochs=hparams.epochs,
        model=hparams.model,
        pretrained=hparams.pretrained,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        # early_stop_callback=early_stopping,
        distributed_backend=hparams.distributed_backend,
        deterministic=True if hparams.seed is not None else False,
        precision=16 if hparams.use_16_bit else 32,
        max_epochs=15,
        min_epochs=5,
        # auto_scale_batch_size='binsearch',
        # auto_lr_find=True,
    )

    if hparams.test:
        trainer.fit(model)
        trainer.test(model)
    elif hparams.load:
        model = WriterID.load_from_checkpoint(hparams.checkpoint_path, vars(hparams))
        trainer.test(model)
    else:
        trainer.fit(model)

    if hparams.use_temp_dir:
        tmp_dir.cleanup()


if __name__ == '__main__':
    main(get_args())
