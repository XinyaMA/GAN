import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory
os.makedirs("generated_progressive_cGAN", exist_ok=True)

# Transform (now loads 64x64 images, but will be resized dynamically during training)
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset
dataset = datasets.ImageFolder(root="C:\\Users\\MA\\Desktop\\master\\S8\\Machine vision\\Newdata\\Train", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Hyperparameters
z_dim = 512  # Increased for progressive training
num_classes = 2
lr = 1e-4
epochs = 800
classes = {0: "cow", 1: "horse"}
num_stages = 7  # 4x4, 8x8, 16x16, 32x32, 64x64, 128x128, 256x256

# ----------------- Progressive Generator -----------------
class ConditionalProgressiveGenerator(nn.Module):
    def __init__(self, z_dim=256, num_classes=2, img_channels=3):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes

        self.label_embed = nn.Embedding(num_classes, z_dim)

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2)
        )

        self.initial_to_rgb = nn.Sequential(
            nn.Conv2d(512, img_channels, 1, 1, 0),
            nn.Tanh()
        )

        self.blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        in_channels = 512

        for _ in range(num_stages - 1):
            out_channels = in_channels // 2
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest')
            ))
            self.to_rgb.append(nn.Sequential(
                nn.Conv2d(out_channels, img_channels, 1, 1, 0),
                nn.Tanh()
            ))
            in_channels = out_channels

        self.current_stage = 0
        self.alpha = 1.0

    def forward(self, noise, labels):
        label_emb = self.label_embed(labels)
        x = (noise * label_emb).view(-1, self.z_dim, 1, 1)
        x = self.initial(x)               
        
        if self.current_stage == 0:
            return self.initial_to_rgb(x)
        # Save the feature map of the previous stage
        x_prev = x
        for i in range(self.current_stage-1):
            x_prev = self.blocks[i](x_prev)
            
        x_current = self.blocks[self.current_stage - 1](x_prev)

        if self.alpha < 1.0:
            if self.current_stage == 1:
                x_old = self.initial_to_rgb(x_prev) 
            else:
                x_old = self.to_rgb[self.current_stage - 2](x_prev)
            x_old = F.interpolate(x_old, scale_factor=2, mode='nearest')
            # current phase RGB
            x_new = self.to_rgb[self.current_stage - 1](x_current) 
            return self.alpha * x_new + (1 - self.alpha) * x_old

        return self.to_rgb[self.current_stage - 1](x_current)

# ----------------- Progressive Discriminator -----------------
class ConditionalProgressiveDiscriminator(nn.Module):
    def __init__(self, num_classes=2, img_channels=3):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 64 * 64)

        # Channel order from high to low
        self.stage_channels = [512, 512, 256, 128, 64, 64, 32]
        self.from_rgb = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.final = nn.ModuleList()

        # Input conversion modules corresponding to each resolution stage
        for ch in self.stage_channels:
            self.from_rgb.append(nn.Sequential(
                nn.Conv2d(img_channels + 1, ch, kernel_size=1),
                nn.LeakyReLU(0.2)
            ))

        # blocks Forward build, from high channel → low channel
        for i in range(len(self.stage_channels) - 1):
            in_ch = self.stage_channels[i]
            out_ch = self.stage_channels[i + 1]
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            ))

        # Final classification layer for each stage
        for ch in self.stage_channels:
            self.final.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, 1, kernel_size=1),
                nn.Sigmoid()
            ))

        self.current_stage = 0
        self.alpha = 1.0

    def forward(self, img, labels):
        batch_size = img.size(0)
        res = img.shape[2]

        label_map = self.label_embed(labels).view(batch_size, 1, 64, 64)
        label_map = F.interpolate(label_map, size=(res, res))
        x = torch.cat([img, label_map], dim=1)

        if self.alpha < 1.0 and self.current_stage > 0:
            downsample = F.avg_pool2d
            x_old_input = downsample(x, 2)
            x_old = self.from_rgb[self.current_stage - 1](x_old_input)  # shape: [B, C_old, H/2, W/2]
            x_old = self.blocks[self.current_stage - 1](x_old)          # 🔁通过 block 升通道+下采样
            x_new = self.from_rgb[self.current_stage](x)                # shape: [B, C_new, H, W]
        
            # Ensure that x_old and x_new are the same size before fusing them together
            if x_old.shape != x_new.shape:
                x_old = F.interpolate(x_old, size=x_new.shape[2:], mode='nearest')
    
            x = self.alpha * x_new + (1 - self.alpha) * x_old
        else:
            x = self.from_rgb[self.current_stage](x)

        out = self.final[self.current_stage](x)
        return out.view(-1)

    
# Initialize models
pgenerator = ConditionalProgressiveGenerator(z_dim=z_dim).to(device)
pdiscriminator = ConditionalProgressiveDiscriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(pgenerator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(pdiscriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training schedule
stage_epochs = [100, 200, 300, 400, 500, 600, 700] #Epochs to transition stages
current_stage = 0

# Training loop
for epoch in range(epochs):
    # Update stage and alpha
    if current_stage < len(stage_epochs)-1 and epoch >= stage_epochs[current_stage]:
        current_stage += 1
        pgenerator.current_stage = current_stage
        pdiscriminator.current_stage = current_stage
        print(f"\nTransitioning to stage {current_stage} ({(4*(2**current_stage))}x{(4*(2**current_stage))})")
    
    # Calculate alpha for smooth transition
    alpha = 1.0
    if current_stage > 0 and epoch < stage_epochs[current_stage]:
        alpha = (epoch - stage_epochs[current_stage-1]) / (stage_epochs[current_stage] - stage_epochs[current_stage-1])
    
    pgenerator.alpha = alpha
    pdiscriminator.alpha = alpha
    
    for real_imgs, labels in dataloader:
        real_imgs, labels = real_imgs.to(device), labels.to(device)
        batch_size = real_imgs.size(0)
        
        # Resize real images to current stage resolution
        current_res = 4 * (2 ** current_stage)
        real_resized = F.interpolate(real_imgs, size=(current_res, current_res))
        
        # --- Train Discriminator ---
        # Real images
        real_pred = pdiscriminator(real_resized, labels)
        real_loss = criterion(real_pred, torch.ones(batch_size, device=device))
        
        # Fake images
        noise = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = pgenerator(noise, labels)
        fake_pred = pdiscriminator(fake_imgs.detach(), labels)
        fake_loss = criterion(fake_pred, torch.zeros(batch_size, device=device))
        
        d_loss = (real_loss + fake_loss) / 2
        
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()
        
        # --- Train Generator ---
        noise = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = pgenerator(noise, labels)
        g_pred = pdiscriminator(fake_imgs, labels)
        g_loss = criterion(g_pred, torch.ones(batch_size, device=device))
        
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()
    
    print(f"[Epoch {epoch+1}/{epochs}] Stage: {current_stage} | Loss_D: {d_loss.item():.4f} | Loss_G: {g_loss.item():.4f}")
    
    # Save samples
    if (epoch + 1) % 100 == 0:
        pgenerator.eval()
        with torch.no_grad():
            for class_id, class_name in classes.items():
                noise = torch.randn(64, z_dim, device=device)
                labels = torch.full((64,), class_id, dtype=torch.long, device=device)
                fake_imgs = pgenerator(noise, labels).cpu()
                grid = utils.make_grid(fake_imgs, nrow=8, normalize=True)
                utils.save_image(grid, f"generated_progressive_cGAN/{class_name}_epoch{epoch+1}_stage{current_stage}.png")
        pgenerator.train()