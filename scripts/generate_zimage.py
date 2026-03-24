import ray, torch, io, os

ray.init(address='192.168.1.41:6379', _temp_dir=r'C:\Users\sarja\AppData\Local\Temp\ray')

# ── STAGE 1: Generate on Laptop 3080 Ti ──────────────────
# Using Stable Diffusion 1.5 — a proper image model for sharp output.
# Z-Image-Turbo was a video model that produced blurry/artifact images.
@ray.remote(num_gpus=1, resources={'node:192.168.1.17': 0.01})
def generate(prompt, negative_prompt, seed):
    import torch, io
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    device = torch.device('cuda')
    print(f'[Stage 1] GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # SD 1.5 = proven, sharp, photorealistic image model (~4 GB in fp16)
    print('Loading Stable Diffusion v1.5...')
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    # DPM++ SDE Karras = sharpest scheduler for SD 1.5
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type='dpmsolver++',
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    print(f'VRAM after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB')

    # Load custom SD 1.5 LoRA
    rank = 16
    lora_path = r'C:\Users\sarja\OneDrive\Desktop\lora_project\output\best_sd_lora.pt'
    print(f'Loading custom LoRA from: {os.path.basename(lora_path)}')
    state = torch.load(lora_path, map_location=device, weights_only=True)
    injected = 0

    for name, module in pipe.unet.named_modules():
        if name not in state:
            continue
        lora_a = state[name]['lora_a'].to(device).to(torch.float16)
        lora_b = state[name]['lora_b'].to(device).to(torch.float16)
        bias   = module.bias
        def make_forward(w, b, la, lb, r=rank):
            def forward(self, x):
                return (torch.nn.functional.linear(x, w, b) +
                        (x @ la.T) @ lb.T * (1.0 / r))
            return forward
        module.forward = make_forward(
            module.weight, bias, lora_a, lora_b).__get__(module, module.__class__)
        injected += 1
    
    print(f'Injected {injected} SD 1.5 LoRA layers!')

    print(f'Seed: {seed}')
    print(f'Prompt: {prompt[:80]}...')
    gen = torch.Generator(device).manual_seed(seed)

    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,      # 25 steps for sharp detail
            guidance_scale=8.0,          # 7.5-9 for architectural photos
            height=512, width=512,
            generator=gen,
        ).images[0]

    buf = io.BytesIO()
    image.save(buf, format='PNG')
    print('[Stage 1] Done!')
    return buf.getvalue()


# ── STAGE 2: ESRGAN upscale on RTX 4090 (no basicsr needed) ──
@ray.remote(num_gpus=1, resources={'node:192.168.1.41': 0.01})
def upscale(img_bytes):
    import torch, io, os, urllib.request
    import torch.nn as nn
    from PIL import Image
    import numpy as np

    device = torch.device('cuda')
    print(f'[Stage 2] Upscaling on: {torch.cuda.get_device_name(0)}')

    # ── Pure PyTorch RRDB architecture (no basicsr) ───────
    class ResidualDenseBlock(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.conv1 = nn.Conv2d(nf,      gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf+gc,   gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf+2*gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf+3*gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf+4*gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
            x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
            x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
            x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(nf, gc)
            self.rdb2 = ResidualDenseBlock(nf, gc)
            self.rdb3 = ResidualDenseBlock(nf, gc)
        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, scale=4):
            super().__init__()
            self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(nf) for _ in range(nb)])
            self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
            self.upconv1   = nn.Conv2d(nf, nf, 3, 1, 1)
            self.upconv2   = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_hr   = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
            self.lrelu     = nn.LeakyReLU(0.2, inplace=True)
        def forward(self, x):
            fea  = self.conv_first(x)
            body = self.conv_body(self.body(fea))
            fea  = fea + body
            fea  = self.lrelu(self.upconv1(
                   torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
            fea  = self.lrelu(self.upconv2(
                   torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
            return self.conv_last(self.lrelu(self.conv_hr(fea)))

    # Download weights if needed
    weights_path = r'C:\Users\WildMindAi\Desktop\lora_project\RealESRGAN_x4plus.pth'
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    if not os.path.exists(weights_path):
        print('Downloading Real-ESRGAN weights (~64MB)...')
        urllib.request.urlretrieve(
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            weights_path)
        print('Downloaded!')

    # Load model
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, scale=4).to(device).half()
    weights = torch.load(weights_path, map_location=device)
    # Handle different checkpoint formats
    state_dict = weights.get('params_ema', weights.get('params', weights))
    # Fix key name mismatch between checkpoint and our architecture
    rename = {
        'conv_up1.weight': 'upconv1.weight', 'conv_up1.bias': 'upconv1.bias',
        'conv_up2.weight': 'upconv2.weight', 'conv_up2.bias': 'upconv2.bias'
    }
    state_dict = {rename.get(k, k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print('ESRGAN model loaded!')

    # Process image in tiles to avoid OOM at 2048x2048
    img    = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t  = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).half().to(device)

    print(f'Input: {img.size} → Upscaling 4x with ESRGAN...')
    tile, pad = 128, 10
    _, _, h, w = img_t.shape
    out_h, out_w = h*4, w*4
    output = torch.zeros(1, 3, out_h, out_w, dtype=torch.float16, device=device)
    count  = torch.zeros(1, 1, out_h, out_w, dtype=torch.float16, device=device)

    with torch.no_grad():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y0 = max(0, y-pad); y1 = min(h, y+tile+pad)
                x0 = max(0, x-pad); x1 = min(w, x+tile+pad)
                patch = img_t[:, :, y0:y1, x0:x1]
                out_p = model(patch).clamp(0, 1)
                # Destination in output
                dy0 = (y-y0)*4; dy1 = dy0 + (y1-y0-(y-y0)-(min(h,y+tile)-y))*4 + (min(h,y+tile)-y)*4
                dx0 = (x-x0)*4; dx1 = dx0 + (x1-x0-(x-x0)-(min(w,x+tile)-x))*4 + (min(w,x+tile)-x)*4
                oy0 = y*4; oy1 = min(out_h, (y+tile)*4)
                ox0 = x*4; ox1 = min(out_w, (x+tile)*4)
                sy0 = (y-y0)*4; sy1 = sy0 + (oy1-oy0)
                sx0 = (x-x0)*4; sx1 = sx0 + (ox1-ox0)
                output[:,:,oy0:oy1,ox0:ox1] += out_p[:,:,sy0:sy1,sx0:sx1]
                count[:,:,oy0:oy1,ox0:ox1]  += 1

    output = (output / count).clamp(0, 1)
    out_np = (output.squeeze(0).permute(1,2,0).cpu().float().numpy() * 255).astype(np.uint8)
    out_img = Image.fromarray(out_np)
    print(f'Output: {out_img.size}')

    buf = io.BytesIO()
    out_img.save(buf, format='PNG')
    print('[Stage 2] Done!')
    return buf.getvalue()


# ── MAIN ─────────────────────────────────────────────────
from PIL import Image
import random, time

# Detailed prompt for sharp, photorealistic temple images
PROMPT = (
    'majestic Indian Hindu temple, grand sandstone architecture, '
    'intricate carved shikhara tower with golden finial, '
    'ornate pillars and arched entrance, saffron flags, '
    'devotees in courtyard, lush green gardens, '
    'golden hour sunlight, dramatic clouds, '
    'aerial view, ultra sharp, highly detailed, '
    'professional architectural photography, 8k, '
    'National Geographic quality, f/8 aperture, HDR'
)

# Strong negative prompt to prevent blur and artifacts
NEGATIVE = (
    'blurry, fuzzy, out of focus, low quality, artifacts, distorted, '
    'watermark, text, ugly, noise, oversaturated, cartoon, painting, '
    'sketch, pixelated, jpeg artifacts, deformed, bad anatomy, '
    'disfigured, low resolution, soft focus, motion blur'
)

SEED = random.randint(1, 999999)
TS   = time.strftime('%H%M%S')

SAVE_512  = os.path.join(
    r'C:\Users\sarja\OneDrive\Desktop\lora_project\output\generated',
    f'temple_{TS}_{SEED}_512.png')
SAVE_2048 = os.path.join(
    r'C:\Users\sarja\OneDrive\Desktop\lora_project\output\generated',
    f'temple_{TS}_{SEED}_2048.png')

print('=' * 60)
print('Temple Image Generation System: SD 1.5 (3080Ti) -> ESRGAN 4x (RTX 4090)')
print('=' * 60)
print(f'Seed: {SEED}')
print(f'Prompt: {PROMPT[:80]}...\n')

print('Stage 1: Generating 512x512 on Laptop (SD 1.5, 25 steps)...')
t0 = time.time()
img_bytes = ray.get(generate.remote(PROMPT, NEGATIVE, SEED))
t1 = time.time()

os.makedirs(os.path.dirname(SAVE_512), exist_ok=True)
Image.open(io.BytesIO(img_bytes)).save(SAVE_512)
print(f'  + 512x512 saved -> {SAVE_512}  ({t1-t0:.1f}s)\n')

print('Stage 2: ESRGAN 4x upscale on RTX 4090...')
t2 = time.time()
upscaled_bytes = ray.get(upscale.remote(img_bytes))
t3 = time.time()

Image.open(io.BytesIO(upscaled_bytes)).save(SAVE_2048)
print(f'  + 2048x2048 saved -> {SAVE_2048}  ({t3-t2:.1f}s)\n')
print(f'\nTotal: {t3-t0:.1f}s  |  Seed: {SEED}')

ray.shutdown()
