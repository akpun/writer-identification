# Writer identification in handwritten text images using deep neural networks   
 

## Requirements  
First, install dependencies   
```bash
# clone project   
git clone https://github.com/akpun/writer-identification.git  

# install requirements through conda   
cd writer-identification 
conda env create -f environment.yml
conda activate WI
 ```   
## Create Datasets
Crop the datasets from the raw original files of [Firemaker (folders p1 and p4)](https://zenodo.org/record/1194612#.XwjRexFw1H6), [IAM (folder forms)](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) and [ICDAR 2017 (folders ScriptNet)](https://zenodo.org/record/1324999#.XwjSMRFw1H4)

```bash
cd datasets
python smart-crop.py dataset-path new-path dataset # dataset choices: firemaker, iam, icdar17
```
For the patches datasets, create from the cropped datasets:
 ```bash
# Firemaker
python create_datasets.py --dataset firemaker --data-path /datasets/crop-firemaker-train/ --test-path /datasets/crop-firemaker-test/ --new-path /datasets/256-100patches-firemaker/ --patch-height 256 --patch-width 256 --num-patches 100

# IAM
python create_datasets.py --dataset iam --data-path /datasets/crop-IAM/ --new-path /datasets/256-100patches-IAM/ --patch-height 256 --patch-width 256  --num-patches 100

# ICDAR17 
python create_datasets.py --dataset icdar17 --data-path /datasets/ScriptNet-HistoricalWI-2017-color/  --new-path 256-100patches-icdar17-binary/ --patch-height 256 --patch-width 256 --num-patches 100

# ICDAR17 binary
python create_datasets.py --dataset icdar17 --binary --data-path /datasets/ScriptNet-HistoricalWI-2017-binarized/  --new-path 256-100patches-icdar17-binary/ --patch-height 256 --patch-width 256 --num-patches 100

```   
For the pages datasets, create from the cropped datasets:
```bash
# Firemaker
python create_datasets.py --pages --dataset firemaker --data-path /datasets/crop-firemaker-train/ --test-path /datasets/crop-firemaker-test/ --new-path /datasets/pages-firemaker/

# IAM
python create_datasets.py --dataset iam --pages --data-path /datasets/crop-IAM/ --new-path /datasets/pages-IAM/

# ICDAR17 
python create_datasets.py --dataset icdar17 --pages --data-path /datasets/ScriptNet-HistoricalWI-2017-binarized/  --new-path /datasets/pages-icdar17/ 

``` 
## Training
Run the following script.   
 ```bash
python writerid.py --dataset iam  --pretrained --data-path /datasets/256-100patches-IAM/ --train-path /datasets/256-100patches-IAM/train --val-path /datasets/256-100patches-IAM/validation --test-path /datasets/256-100patches-IAM/test

# Flags:
# Batch size and learning rate, default 32 and 0.01
python writerid.py  --batch-size 16 --lr 0.002

# Use specific seed 
python writerid.py --seed 420

# Use 16 bit precision, need nvidia apex installed
python writerid.py --use-16-bit

# CPU   
python writerid.py     

# MULTIPLE GPUS
python writerid.py --gpus 4

# SPECIFIC GPUs
python writerid.py --gpus '0,3'

# Create dataset in temporary directory specifying patch size and number of patches, and train in this dataset
python writerid.py  --dataset firemaker --use-temp-dir  --num-patches 100 --patch-height 256 --pretrained --data-path /datasets/crop-firemaker-train/ --test-path /datasets/crop-firemaker-test/

# Use a model of any of the pretrained models of torchvision
python writerid.py  --dataset firemaker --model resnet34  --pretrained 

```   
## Evaluation
 ```bash
# Test at the end of training add --test flag
python writerid.py --test --dataset iam  --pretrained --data-path /datasets/256-100patches-IAM/ --train-path /datasets/256-100patches-IAM/train --val-path /datasets/256-100patches-IAM/validation --test-path /datasets/256-100patches-IAM/test

# Test from pretrained model add --load flag and specify path of model in --checkpoint-path
python writerid.py --dataset iam --load  --test-path /datasets/256-100patches-IAM/test --checkpoint-path /checkpoints/iam_resnet18_epoch=09-val_loss=0.31.ckpt

 ```


