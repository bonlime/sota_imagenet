# Evaluation results of pretrainde models:
` python3 train.py -a resnet18 --phases="[{'ep':0,'sz':224,'bs':128,'rect_val':False, 'lr':0,'mom':0}]" --pretrained -e -p 100`
Values in brackets are what we need. Values before are metrics in the last batch
* Resnet18: NoRect, 224:             Loss 0.3903 (1.2497)    Acc@1 92.188 (69.656)   Acc@5 97.656 (88.986)
* Resnet18: Rect, 224:                Loss 1.5552 (1.2140)    Acc@1 61.719 (70.686)   Acc@5 85.156 (89.702)
* Resnet18: Rect, 256:                Loss 1.1446 (1.2116)    Acc@1 73.438 (71.338)   Acc@5 92.969 (90.248) 
* Mobilenetv2: NoRect, 224            Loss 0.9992 (1.1483)    Acc@1 73.438 (71.704)   Acc@5 91.406 (90.360)
* Mobilenetv2: Rect, 256              Loss 1.1397 (1.1010)    Acc@1 69.531 (73.464)   Acc@5 93.750 (91.686)


Exp1:
Resnet50, 50 epochs, the same phases. 
* SGD - trained to 74%. Less than Sota. Probably due to suboptimal LR
* SGDW - Run Falied due to loss scaling
* SGDW + Nesterov - run failed
* SGD + Nesterov - runs 2 times slower that SGD

Exp2: increased bs to proper one + loss_scale=2048
* SGD - better accucary
* SGDW - Fail
* SGD + no_bn_wd - accuracy almost the same, but loss is lower (!)
* PMSProp - Fail 

Exp3:
* Mobilenetv2 as in original paper, without any modifications for 90 epochs
 
 python3 train.py --load-phases -a mobilenet_v2 -p 500 --opt_level=O2 --gpu=1 --logdir=logs/mbln2_rms-tmp --wd=4e-5  --optim=rmsprop --optim-param="{'alpha':0.9}" 




Code to reproduce ImageNet in 18 minutes, by Andrew Shaw, Yaroslav Bulatov, and Jeremy Howard. High-level overview of techniques used is [here](http://fast.ai/2018/08/10/fastai-diu-imagenet/)


Pre-requisites: Python 3.6 or higher

```
pip install -r requirements.txt
aws configure  (or set your AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_DEFAULT_REGION)
python train.py  # pre-warming
python train.py 
```

To run with smaller number of machines:

```
python train.py --machines=1
python train.py --machines=4
python train.py --machines=8
python train.py --machines=16
```

Your AWS account needs to have high enough limit in order to reserve this number of p3.16xlarge instances. The code will set up necessary infrastructure like EFS, VPC, subnets, keypairs and placement groups. Therefore permissions for these those resources are needed.


# Checking progress

Machines print progress to local stdout as well as logging TensorBoard event files to EFS. You can:

1. launch tensorboard using tools/launch_tensorboard.py

That will provide a link to tensorboard instance which has loss graph under "losses" group. You'll see something like this under "Losses" tab
<img src='https://raw.githubusercontent.com/diux-dev/imagenet18/master/tensorboard.png'>

2. Connect to one of the instances using instructions printed during launch. Look for something like this

```
2018-09-06 17:26:23.562096 15.imagenet: To connect to 15.imagenet
ssh -i /Users/yaroslav/.ncluster/ncluster5-yaroslav-316880547378-us-east-1.pem -o StrictHostKeyChecking=no ubuntu@18.206.193.26
tmux a
```

This will connect you to tmux session and you will see something like this

```
.997 (65.102)   Acc@5 85.854 (85.224)   Data 0.004 (0.035)      BW 2.444 2.445
Epoch: [21][175/179]    Time 0.318 (0.368)      Loss 1.4276 (1.4767)    Acc@1 66.169 (65.132)   Acc@5 86.063 (85.244)   Data 0.004 (0.035)      BW 2.464 2.466
Changing LR from 0.4012569832402235 to 0.40000000000000013
Epoch: [21][179/179]    Time 0.336 (0.367)      Loss 1.4457 (1.4761)    Acc@1 65.473 (65.152)   Acc@5 86.061 (85.252)   Data 0.004 (0.034)      BW 2.393 2.397
Test:  [21][5/7]        Time 0.106 (0.563)      Loss 1.3254 (1.3187)    Acc@1 67.508 (67.693)   Acc@5 88.644 (88.315)
Test:  [21][7/7]        Time 0.105 (0.432)      Loss 1.4089 (1.3346)    Acc@1 67.134 (67.462)   Acc@5 87.257 (88.124)
~~21    0.31132         67.462          88.124
```

The last number indicates that at epoch 21 the run got 67.462 top-1 test accuracy and 88.124 top-5 test accuracy.
