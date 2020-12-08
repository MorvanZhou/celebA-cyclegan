An implementation of cycle-gan that trains on [celebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Using wgan with gradient penalty to stabilise training and avoid model collapse.

## Run code
install requirements:
```shell script
git clone https://github.com/MorvanZhou/celebA-cyclegan
cd celebA-cyclegan
pip3 install -r requirements.txt
```

Download CelebA Align&Cropped Images from [https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8](https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8),
and Attributes Annotations from [https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8](https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8).

Parse data
```shell script
python3 dataset.py --data_dir D:/data/celebA_img_align/
```

Training
```shell script
python3 train.py --data_dir D:/data/celebA_img_align/ -b 5 --epoch 101 --cycle_lambda 10 --gp_lambda 10 -lr 0.001 -b1 0.1 -b2 0.99
```

![ep003t7500.png](/demo/ep003t7500.png)
