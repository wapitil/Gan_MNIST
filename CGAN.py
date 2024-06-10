import jittor as jt
from jittor import init
import argparse
import os
import numpy as np
import math
from jittor import nn
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams

class CGAN:
    def __init__(self, opt):
        self.opt = opt
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.adversarial_loss = nn.MSELoss()
        self.optimizer_G = nn.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = nn.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.d_losses = []  # 记录 D loss
        self.g_losses = []  # 记录 G loss
        self.epochs = []

    def build_generator(self):
        class Generator(nn.Module):
            def __init__(self, opt, img_shape):
                super(Generator, self).__init__()
                self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
                self.img_shape = img_shape
                def block(in_feat, out_feat, normalize=True):
                    layers = [nn.Linear(in_feat, out_feat)]
                    if normalize:
                        layers.append(nn.BatchNorm1d(out_feat, 0.8))
                    layers.append(nn.LeakyReLU(0.2))
                    return layers
                self.model = nn.Sequential(
                    *block(opt.latent_dim + opt.n_classes, 128, normalize=False), 
                    *block(128, 256), 
                    *block(256, 512), 
                    *block(512, 1024), 
                    nn.Linear(1024, int(np.prod(self.img_shape))), 
                    nn.Tanh()
                )

            def execute(self, noise, labels):
                gen_input = jt.contrib.concat((self.label_emb(labels), noise), dim=1)
                img = self.model(gen_input)
                img = img.view((img.shape[0], *self.img_shape))
                return img

        return Generator(self.opt, self.img_shape)

    def build_discriminator(self):
        class Discriminator(nn.Module):
            def __init__(self, opt, img_shape):
                super(Discriminator, self).__init__()
                self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
                self.img_shape = img_shape
                self.model = nn.Sequential(
                    nn.Linear(opt.n_classes + int(np.prod(self.img_shape)), 512), 
                    nn.LeakyReLU(0.2), 
                    nn.Linear(512, 512), 
                    nn.Dropout(0.4), 
                    nn.LeakyReLU(0.2), 
                    nn.Linear(512, 512), 
                    nn.Dropout(0.4), 
                    nn.LeakyReLU(0.2), 
                    nn.Linear(512, 1)
                )

            def execute(self, img, labels):
                d_in = jt.contrib.concat((img.view((img.shape[0], -1)), self.label_embedding(labels)), dim=1)
                validity = self.model(d_in)
                return validity

        return Discriminator(self.opt, self.img_shape)

    def save_image(self, img, path, nrow=10, padding=5):
        N, C, W, H = img.shape
        if N % nrow != 0:
            print("N%nrow!=0")
            return
        ncol = int(N / nrow)
        img_all = []
        for i in range(ncol):
            img_ = []
            for j in range(nrow):
                img_.append(img[i*nrow + j])
                img_.append(np.zeros((C, W, padding)))
            img_all.append(np.concatenate(img_, 2))
            img_all.append(np.zeros((C, padding, img_all[0].shape[2])))
        img = np.concatenate(img_all, 1)
        img = np.concatenate([np.zeros((C, padding, img.shape[2])), img], 1)
        img = np.concatenate([np.zeros((C, img.shape[1], padding)), img], 2)
        min_ = img.min()
        max_ = img.max()
        img = (img - min_) / (max_ - min_) * 255
        img = img.transpose((1, 2, 0))
        if C == 3:
            img = img[:, :, ::-1]
        elif C == 1:
            img = img[:, :, 0]
        Image.fromarray(np.uint8(img)).save(path)

    def sample_image(self, n_row, batches_done):
        z = jt.array(np.random.normal(0, 1, (n_row ** 2, self.opt.latent_dim))).float32().stop_grad()
        labels = jt.array(np.array([num for _ in range(n_row) for num in range(n_row)])).float32().stop_grad()
        gen_imgs = self.generator(z, labels)
        self.save_image(gen_imgs.numpy(), "./imgs/%d.png" % batches_done, nrow=n_row)

    def train(self, dataloader):
        for epoch in range(self.opt.n_epochs):
            dataloader.set_attrs(shuffle=True)

            for i, (imgs, labels) in enumerate(dataloader):
                batch_size = imgs.shape[0]

                valid = jt.ones([batch_size, 1]).float32().stop_grad()
                fake = jt.zeros([batch_size, 1]).float32().stop_grad()

                real_imgs = jt.array(imgs)
                labels = jt.array(labels)

                z = jt.array(np.random.normal(0, 1, (batch_size, self.opt.latent_dim))).float32()
                gen_labels = jt.array(np.random.randint(0, self.opt.n_classes, batch_size)).float32()

                gen_imgs = self.generator(z, gen_labels)
                validity = self.discriminator(gen_imgs, gen_labels)
                g_loss = self.adversarial_loss(validity, valid)
                g_loss.sync()
                self.optimizer_G.step(g_loss)

                validity_real = self.discriminator(real_imgs, labels)
                d_real_loss = self.adversarial_loss(validity_real, valid)

                validity_fake = self.discriminator(gen_imgs.stop_grad(), gen_labels)
                d_fake_loss = self.adversarial_loss(validity_fake, fake)

                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.sync()
                self.optimizer_D.step(d_loss)

            # 在每个epoch结束时记录损失值并生成图像
            self.d_losses.append(d_loss.data.item())
            self.g_losses.append(g_loss.data.item())
            self.sample_image(n_row=10, batches_done=epoch)

            print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, self.opt.n_epochs, d_loss.data, g_loss.data)
            )

            if epoch % 10 == 0:
                self.generator.save("./models/generator_last.pkl")
                self.discriminator.save("./models/discriminator_last.pkl")

        # 绘制损失曲线
        self.plot_losses()

    def plot_losses(self):
        config = {
            "font.family": 'serif',
            "font.size": 18,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
        }
        rcParams.update(config)
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label='判别器损失')
        plt.plot(self.g_losses, label='生成器损失')
        plt.xlabel('周期')
        plt.ylabel('损失')
        plt.legend()
        plt.title('训练损失')
        plt.show()

    def generate(self, number):
        self.generator.eval()
        self.generator.load('./models/generator_last.pkl')

        n_row = len(number)
        z = jt.array(np.random.normal(0, 1, (n_row, self.opt.latent_dim))).float32().stop_grad()
        labels = jt.array(np.array([int(num) for num in number])).float32().stop_grad()
        gen_imgs = self.generator(z, labels)

        img_array = gen_imgs.data.transpose((1, 2, 0, 3))[0].reshape((gen_imgs.shape[2], -1))
        min_ = img_array.min()
        max_ = img_array.max()
        img_array = (img_array - min_) / (max_ - min_) * 255
        Image.fromarray(np.uint8(img_array)).save("result.png")
        return "result.png"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=150, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
    parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
    parser.add_argument('--number', type=str, default='123', help='Generate input number pictures')
    opt = parser.parse_args()

    if jt.has_cuda:
        jt.flags.use_cuda = 1

    cgan = CGAN(opt)
    # 加载MNIST数据集
    from jittor.dataset.mnist import MNIST
    import jittor.transform as transform    
    transform = transform.Compose([
        transform.Resize(opt.img_size),
        transform.Gray(),
        transform.ImageNormalize(mean=[0.5], std=[0.5]),
    ])
    

    # 如果需要训练模型则取消下列两行注释
    # dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)
    # cgan.train(dataloader) 

    # 生成图像并进行评估
    result_path = cgan.generate(opt.number)
    print(f"Generated image saved at {result_path}")
    return result_path

if __name__ == "__main__":
    main()

