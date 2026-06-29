import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import logging
from tqdm import tqdm
import random

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from src.shared.tennis_dataset import TennisSkeletonDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=9):
        super().__init__()
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def main_worker(gpu, ngpus_per_node, args):
    is_main_process = (gpu == 0)
    
    if ngpus_per_node > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:23456',
            world_size=ngpus_per_node,
            rank=gpu
        )
    
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")

    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    start_epoch = 0
    if args.resume and os.path.exists(args.checkpoint_path):
        if is_main_process:
            logging.info(f"Resuming from checkpoint {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['G'])
        discriminator.load_state_dict(checkpoint['D'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Wrap models with DDP
    if ngpus_per_node > 1:
        generator = DDP(generator, device_ids=[gpu])
        discriminator = DDP(discriminator, device_ids=[gpu])

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    if args.resume and os.path.exists(args.checkpoint_path):
        optimizer_G.load_state_dict(checkpoint['opt_G'])
        optimizer_D.load_state_dict(checkpoint['opt_D'])

    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_pixelwise = nn.L1Loss().to(device)

    transform_pipeline = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size * 3), Image.BICUBIC),
        transforms.ToTensor(),
    ])

    dataset = TennisSkeletonDataset(args.data_root, subset=args.subset, transform=transform_pipeline)

    if ngpus_per_node > 1:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    # Resolve batch size, defaulting to 64 if 'auto' was passed to better utilize 48GB VRAM
    batch_size = 64 if str(args.batch_size).lower() == "auto" else int(args.batch_size)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(sampler is None), 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        persistent_workers=(args.num_workers > 0)
    )

    if is_main_process:
        logging.info(f"Starting training on {len(dataset)} images for {args.epochs} epochs with batch size {batch_size} (per GPU)")

    if len(dataloader) == 0:
        if is_main_process:
            logging.error(f"Dataloader is empty! Dataset size ({len(dataset)}) is too small for batch size {batch_size} across {ngpus_per_node} GPUs with drop_last=True.")
        return

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if is_main_process:
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{args.epochs}")
            iterator = pbar
        else:
            iterator = enumerate(dataloader)

        for i, (real_A, real_B) in iterator:
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            valid = torch.ones((real_A.size(0), 1, real_A.size(2) // 16, real_A.size(3) // 16), device=device, requires_grad=False)
            fake = torch.zeros((real_A.size(0), 1, real_A.size(2) // 16, real_A.size(3) // 16), device=device, requires_grad=False)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Generate fake image conditioned on skeleton and reference image
            gen_input = torch.cat((real_A, ref_img), 1)
            fake_B = generator(gen_input)
            
            # In DDP, calling a module multiple times before backward() can cause hook/graph collisions.
            # Combine real and fake into a single forward pass:
            combined_A = torch.cat((gen_input, gen_input), 0)
            combined_B = torch.cat((real_B, fake_B.detach()), 0)
            pred_combined = discriminator(combined_A, combined_B)
            
            pred_real, pred_fake_detached = torch.chunk(pred_combined, 2, dim=0)
            
            # Real loss
            loss_real = criterion_GAN(pred_real, valid)
            
            # Fake loss
            loss_fake = criterion_GAN(pred_fake_detached, fake)
            
            # Total D loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # G needs to fool D, so we evaluate D(fake_B) with the UPDATED discriminator weights
            pred_fake = discriminator(gen_input, fake_B)
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            
            # Total G loss
            loss_G = loss_GAN + args.lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            if is_main_process:
                pbar.set_postfix({"D_loss": f"{loss_D.item():.4f}", "G_loss": f"{loss_G.item():.4f}"})

        if is_main_process:
            sample_img = torch.cat((real_A[:4], ref_img[:4], real_B[:4], fake_B[:4]), -1)
            vutils.save_image(sample_img, f"{args.sample_dir}/epoch_{epoch:03d}.png", nrow=4, normalize=True)

            if (epoch + 1) % 10 == 0:
                state_dict_G = generator.module.state_dict() if ngpus_per_node > 1 else generator.state_dict()
                state_dict_D = discriminator.module.state_dict() if ngpus_per_node > 1 else discriminator.state_dict()
                
                torch.save(state_dict_G, args.out_weights)
                torch.save({
                    'epoch': epoch,
                    'G': state_dict_G,
                    'D': state_dict_D,
                    'opt_G': optimizer_G.state_dict(),
                    'opt_D': optimizer_D.state_dict(),
                }, args.checkpoint_path)

    if is_main_process:
        state_dict_G = generator.module.state_dict() if ngpus_per_node > 1 else generator.state_dict()
        torch.save(state_dict_G, args.out_weights)
        logging.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train Pix2Pix for Pointstream GenAI Backend")
    parser.add_argument("--data-root", type=str, default="assets/dataset/pix2pix", help="Path to pix2pix dataset root")
    parser.add_argument("--subset", type=str, default="all", help="Subset folder name inside data-root or 'all' to use all subsets")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=str, default="auto", help="Batch size per GPU ('auto' maps to 64)")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 4), help="Number of CPU workers for data loading")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--lambda-pixel", type=float, default=100)
    parser.add_argument("--out-weights", type=str, default="assets/weights/pix2pix_generator.pt")
    parser.add_argument("--checkpoint-path", type=str, default="assets/weights/pix2pix_checkpoint.pt")
    parser.add_argument("--sample-dir", type=str, default="assets/samples")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_weights), exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        logging.info(f"Using {ngpus_per_node} GPUs with DistributedDataParallel (DDP)!")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(0, 1, args)

if __name__ == "__main__":
    main()
