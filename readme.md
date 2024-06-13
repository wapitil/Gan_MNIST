## Introduction
本项目使用数字图片数据集 MNIST，训练一个将随机噪声和类别标签映射为数字图片的Conditional GAN模型，并生成指定数字序列。同时，本项目另外训练了一个识别生成图片并返回识别结果的模型，以完成项目需要。
生成的图片如下所示:
<div align="center">
    <img src="https://github.com/wapitil/Gan_MNIST/blob/main/docx_img/0_150.png" alt="Generated Image" style="width: 75%;">
</div>
以下为项目主要文件:

- recognize.py
- CGAN.py
- mnist_classify.py

### 详细说明
- recognize.py 手写数字识别程序，运行该命令，能够识别并返回识别后的数字结果
- CGAN.py 为训练图像，生成图像的代码。
1. 如果您想训练自己的模型，请将CGAN.py中下面两行代码解开注释，进行训练
```python
dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)
cgan.train(dataloader) 
```
2. 如果您不想进行训练，本项目在models 目录下提供了 **generator_last.pkl** 文件。
```
python ./CGAN --number 213123
```
在终端输入上列命令，您将看到一个名为result.png。其中 213123 只是示例输入，您可以在这里输入任何您想要生成的数字。
<br> </br>
![img](https://github.com/wapitil/Gan_MNIST/blob/main/result.png)

- mnist_classify.py 使用下列命令运行该文件，您可以生成自己的数字分类模型。如果您不想进行训练，本项目在models 目录下提供了mnist_cnn.pt 以供使用。




