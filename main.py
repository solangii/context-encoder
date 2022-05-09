from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import glob
import pandas
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os
from utils import *

device = 'cuda:0'


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about execute option
    parser.add_argument('--dataroot', type=str, default='data/train')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=1)

    # about training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)

    # about test option
    parser.add_argument('--weight_path', type=str, default='model.pth', help='used for test')
    parser.add_argument('--threshold', type=float, default=0.2, help='using at calculating difference')

    # about save
    parser.add_argument('--save_root', type=str, default='../result/')
    parser.add_argument('--memo', type=str, default='',
                        help='make folder with value of parameter at `result/[dataset]/img`')

    config = parser.parse_args()

    return config

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class FaceDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, method="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = method
        self.root = root
        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = self.files[:-4000] if self.mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1
        return masked_img, masked_part

    def apply_center_mask(self, img):

        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1
        return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            masked_img, aux = self.apply_random_mask(img)
        else:
            masked_img, aux = self.apply_center_mask(img)
        return img, masked_img, aux

    def __len__(self):
        return len(self.files)

class Trainer(object):
    def __init__(self, config):
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.lr
        self._build_model()
        self.binary_cross_entropy = torch.nn.BCELoss()
        self.p_loss = torch.nn.L1Loss()
        self.adv_loss = torch.nn.MSELoss()
        transforms_ = [transforms.Resize((128, 128), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataset = FaceDataset(root=config.dataroot, method=config.mode, transforms_=transforms_)
        self.root = dataset.root
        
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer_G = torch.optim.Adam(self.gnet.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.dnet.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        print("Training...")
    
    def _build_model(self):
        gnet, dnet = Generator(), Discriminator()
        self.gnet = gnet.to(device)
        self.dnet = dnet.to(device)
        self.gnet.apply(weights_init_normal)
        self.dnet.apply(weights_init_normal)
        self.gnet.train()
        self.dnet.train()

        print('Finish build model.')

    def train(self, config):
        for epoch in tqdm.tqdm(range(self.epochs + 1)):
            if epoch % 10 == 0:
                torch.save(self.gnet.state_dict(), "_".join([config.save_root, str(epoch), '.pth'])) #Change this path

            for batch_idx, (imgs, masked_imgs, masked_parts) in enumerate(self.dataloader):
                Tensor = torch.cuda.FloatTensor
                patch_h, patch_w = int(64 / 2 ** 3), int(64 / 2 ** 3)
                patch = (1, patch_h, patch_w)

                # adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

                # make input
                imgs = Variable(imgs.type(Tensor))
                masked_imgs = Variable(masked_imgs.type(Tensor))
                masked_parts = Variable(masked_parts.type(Tensor))

                # Train Generator

                self.optimizer_G.zero_grad()
                gen_parts = self.gnet(masked_imgs)

                pixel = self.p_loss(gen_parts, masked_parts)
                adv = self.adv_loss(self.dnet(gen_parts), valid)
                g_loss = pixel + 0.0001 * adv
                g_loss.backward()
                self.optimizer_G.step()

                # Train Discriminator

                self.optimizer_D.zero_grad()

                real_loss = self.adv_loss(self.dnet(masked_parts), valid)
                fake_loss = self.adv_loss(self.dnet(gen_parts.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                self.optimizer_D.step()

                print("[Epoch %d/%d] [Batch %d/%d]" % (epoch, self.epochs+1, batch_idx, len(self.dataloader)))
             

class Tester(object):
    def __init__(self, config):
        self._build_model()
        transforms_ = [transforms.Resize((128, 128), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        dataset = FaceDataset(root=config.dataroot, method=config.mode, transforms_=transforms_)
        self.root = dataset.root        
        self.test_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        print("Testing...")

    def _build_model(self):
        gnet = Generator()
        self.gnet = gnet.to(device)
        self.gnet.load_state_dict(torch.load(config.weight_path))
        self.gnet.eval()
        print('Finish build model.')

    def test(self, config):
        Tensor = torch.cuda.FloatTensor
        samples, masked_samples, i = next(iter(self.test_dataloader))
        samples = Variable(samples.type(Tensor))
        masked_samples = Variable(masked_samples.type(Tensor))
        i = i[0].item()  # Upper-left coordinate of mask

        # Generate inpainted image
        gen_mask = self.gnet(masked_samples)
        filled_samples = masked_samples.clone()
        filled_samples[:, :, i : i + 64, i : i + 64] = gen_mask

        # Save sample
        sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
        save_image(sample, "img/test.png", nrow=6, normalize=True)   #Change this path

def main():
    config = get_command_line_parser()
    set_seed(config.seed)  # for reproduction
    config.save_path = set_save_path(config)

    if config.mode == 'train':
        print("train mode!")
        trainer = Trainer(config)
        trainer.train(config)
    else:
        print("test mode!")
        tester = Tester(config)
        tester.test(config)


if __name__ == '__main__':
    main()
