import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory
os.makedirs("generated_cGAN_wgan", exist_ok=True)

# Transform
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset: Train contains Cow/ and Horse/ folders
dataset = datasets.ImageFolder(root="C:\\Users\\MA\\Desktop\\master\\S8\\Machine vision\\Newdata\\Train", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Hyperparameters
z_dim = 32
num_classes = 2
lr = 1e-4
epochs = 3000
classes = {0: "cow", 1: "horse"}
lambda_gp = 10 # Gradient Penalty
n_critic = 2   # Number of discriminator updates vs. 1 for generator

# ----------------- Generator -----------------
class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim=200, img_channels=3, feature_g=64, num_classes=2):
        super().__init__()
        self.z_dim = z_dim
        self.label_embed = nn.Embedding(num_classes, z_dim)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8), nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4), nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2), nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g), nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_embed(labels)
        x = noise * label_emb
        x = x.view(-1, self.z_dim, 1, 1)
        return self.net(x)

# ----------------- Discriminator -----------------
class ConditionalDiscriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64, num_classes=2):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 64 * 64)

        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_d * 2),  # 改用InstanceNorm
            nn.LeakyReLU(0.2),
            # 新增层
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_d * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, img, labels):
        batch_size = img.size(0)
        label_map = self.label_embed(labels).view(batch_size, 1, 64, 64)
        x = torch.cat([img, label_map], dim=1)
        return self.net(x).view(-1)


# WGAN-GP function
def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    batch_size = real_samples.size(0)
    # 生成随机插值系数
    alpha = torch.rand(batch_size, 1, 1, 1).to(real_samples.device)
    # 创建插值样本
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    # 计算判别器对插值样本的输出
    d_interpolates = D(interpolates, labels)
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    # 计算梯度惩罚项
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Initialize models
cgenerator = ConditionalGenerator(z_dim=z_dim).to(device)
cdiscriminator = ConditionalDiscriminator().to(device)

# Loss and optimizers
optimizer_g = optim.Adam(cgenerator.parameters(), lr=lr, betas=(0.0, 0.99))
optimizer_d = optim.Adam(cdiscriminator.parameters(), lr=1e-4, betas=(0.0, 0.99))

# Training loop
losses_g_cGAN = []
losses_d_cGAN = []

scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=100, gamma=0.95)
scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=100, gamma=0.95)

for epoch in range(epochs):
    for real_imgs, labels in dataloader:
        real_imgs, labels = real_imgs.to(device), labels.to(device)
        batch_size = real_imgs.size(0)

        # ===== 训练判别器 =====
        for _ in range(n_critic):
            # 生成假样本
            noise = torch.randn(batch_size, z_dim, device=device)
            fake_imgs = cgenerator(noise, labels)
            
            # 计算判别器损失
            d_real = cdiscriminator(real_imgs, labels)
            d_fake = cdiscriminator(fake_imgs.detach(), labels)
            loss_d = -torch.mean(d_real) + torch.mean(d_fake)
            
            # 计算梯度惩罚
            gp = compute_gradient_penalty(cdiscriminator, real_imgs, fake_imgs.data, labels)
            loss_d += lambda_gp * gp
            
            # 更新判别器
            cdiscriminator.zero_grad()
            loss_d.backward()
            optimizer_d.step()
        
        # ===== 训练生成器 =====
        # 生成假样本
        noise = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = cgenerator(noise, labels)
        
        # 计算生成器损失
        d_fake = cdiscriminator(fake_imgs, labels)
        loss_g = -torch.mean(d_fake)
        
        # 更新生成器
        cgenerator.zero_grad()
        loss_g.backward()
        optimizer_g.step()

    scheduler_g.step()
    scheduler_d.step()

    losses_g_cGAN.append(loss_g.item())
    losses_d_cGAN.append(loss_d.item())

    print(f"[Epoch {epoch+1}/{epochs}] Loss_D: {loss_d.item():.4f} | Loss_G: {loss_g.item():.4f}")

    # Save different cow/horse images (from different z) every 100 epochs
    if (epoch + 1) % 1000 == 0:
        cgenerator.eval()
        with torch.no_grad():
            for class_id, class_name in classes.items():
                z = torch.randn(64, z_dim, device=device)
                labels = torch.full((64,), class_id, dtype=torch.long, device=device)
                fake_imgs = cgenerator(z, labels).cpu()
                grid = utils.make_grid(fake_imgs, nrow=8, normalize=True)
                utils.save_image(grid, f"generated_cGAN_wgan/{class_name}_epoch{epoch+1}.png")

plt.figure(figsize=(10, 5))
plt.plot(losses_g_cGAN, label='Generator Loss')
plt.plot(losses_d_cGAN, label='Discriminator Loss')
plt.title("Generator and Discriminator Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()