# Persistence of Neural Network Watermarks
This is the code for our paper "Persistence of Backdoor-based Watermarks for Neural Networks: A Comprehensive Evaluation", which is accepted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS). Please refer to our full paper [here](https://arxiv.org/abs/2501.02704).

## How to run the code
To run end-to-end: 
```shell
python main.py <PATH-TO-CONFIG>
```
, where `<PATH-TO-CONFIG>` is the location of YAML config file in directory `cfg`, e.g. `cfg/adi_resnet_cifar10_noise.yml`.

The end-to-end running executes:

1. Data generation (if data directory is not available)
2. Model training and watermark embedding
3. Model finetuning with extended data
4. Model retraining with training data

Each phase of the program can also be run as a standalone code, namely:

1. Data generation: `python data_utils.py <dataset> <data_path> <trigger_type> --noise 0.2 --new_label 0 --trigger_size 200 --finetune_size 0.3`, where 
* `<dataset>` is `cifar10`
* `<data_path>` is `data/CIFAR10`
* `<trigger_type>` is either `unrelated, textoverlay, noise, adv`
* `noise` is the intensity of Gaussian noise for trigger type noise, default is 1.0
* `new_label` is the label assigned to trigger samples, can be `None`, or an integer
* `trigger_size` is the number of trigger samples
* `finetune_size` is ratio of finetune data to the whole dataset
2. Model training and watermark embedding: `python train_initial.py cfg/initial/<CONFIG-FILE>`
3. Model finetuning: `python finetune.py cfg/finetune/<CONFIG-FILE>`
4. Model retraining: `python retrain.py cfg/retrain.py cfg/retrain/<CONFIG-FILE>`

## Important configs

Each config contains some config, belows are some important ones:
```yaml
dataset: cifar10
data_path: data/CIFAR10
trigger_type: either noise, unrelated, textoverlay or adv
trigger_label: single or multiple
new_label: the label of trigger samples, default 0, this will be ignored if trigger_label = multiple
trigger_size: an int
finetune_size: ratio of finetune
model: ResNet18, ResNetCBN18
save_dir: checkpoints
save_name: name of the saved model
exp_name: name of experiment, ignore if not use wandb
method: adi, app, rowback or certified
batchsize: an int
batchsize_wm: an int
batchsize_c: an int, ignored if method is NOT app
optimizer: a single string, either sgd, adam. Must be a list of 3 if run end-to-end, e.g. [sgd, adam, sgd]
lr: a float in single phase or a list of 3 floats in end-to-end
wd: same settings as lr
scheduler: single string from step, multistep, cosine, na. List of 3 strings in end-to-end
scheduler_step: an int or list of 3 ints
scheduler_gamma: a float or list of 3 floats
epochs: an int or list of 3 ints
eval_robust: evaluate median/avg wm accuracies, should be True if method = certified
log_wandb: True or False
seed: an int
```