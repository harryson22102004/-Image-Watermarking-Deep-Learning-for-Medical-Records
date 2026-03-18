import torch, torch.nn as nn
import numpy as np
 
class WatermarkEncoder(nn.Module):
    def __init__(self, msg_len=64):
        super().__init__()
        self.msg_proj = nn.Sequential(nn.Linear(msg_len, 64*64), nn.ReLU())
        self.embed = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1), nn.Tanh())
    def forward(self, img, msg):
        msg_map = self.msg_proj(msg).view(-1, 1, 64, 64)
        msg_map = msg_map.expand(-1, 1, img.shape[2], img.shape[3])
        return img + 0.1*self.embed(torch.cat([img, msg_map], dim=1))
 
class WatermarkDecoder(nn.Module):
    def __init__(self, msg_len=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU()
      nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, msg_len), nn.Sigmoid())
    def forward(self, x): return self.net(x)
 
def psnr(original, watermarked):
    mse = nn.MSELoss()(original, watermarked).item()
    return 10 * np.log10(1.0 / (mse + 1e-10))
 
def bit_accuracy(msg_orig, msg_decoded, thresh=0.5):
    pred = (msg_decoded > thresh).float()
    return (pred == msg_orig).float().mean().item()
 
encoder = WatermarkEncoder(64); decoder = WatermarkDecoder(64)
img = torch.rand(2, 3, 256, 256)
msg = torch.randint(0, 2, (2, 64)).float()
watermarked = encoder(img, msg)
recovered = decoder(watermarked)
p = psnr(img, watermarked)
acc = bit_accuracy(msg, recovered)
print(f"PSNR: {p:.1f} dB (>35dB = imperceptible)")
print(f"Bit accuracy: {acc:.3f} (1.0 = perfect recovery)")
print(f"Watermark distortion: {(watermarked-img).abs().mean().item():.4f}")
