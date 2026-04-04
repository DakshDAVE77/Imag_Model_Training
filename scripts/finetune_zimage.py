import ray, torch, os

ray.init(address='192.168.1.31:6379', _temp_dir=r'C:\Users\sarja\AppData\Local\Temp\ray')

@ray.remote(num_gpus=0.5, resources={'node:192.168.1.31': 0.01})
def preprocess_images(dataset_path):
    from PIL import Image
    from torchvision import transforms
    import os

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    images, captions = [], []
    files = sorted([f for f in os.listdir(dataset_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f'Using {len(files)} images')
    for file in files:
        img = Image.open(os.path.join(dataset_path, file)).convert('RGB')
        images.append(transform(img).numpy().tolist())
        txt = os.path.splitext(os.path.join(dataset_path, file))[0] + '.txt'
        captions.append(open(txt).read().strip() if os.path.exists(txt)
                        else 'a photo of sks temple')
    print(f'Preprocessed {len(images)} images')
    return images, captions


@ray.remote(num_gpus=1, resources={'node:192.168.1.31': 0.01}, max_retries=0)
def finetune_lora(images, captions, output_dir, num_epochs=50, lr=3e-4, rank=8):
    import torch, os, gc, numpy as np, time
    from diffusers import ZImagePipeline
    from torch.optim import AdamW

    device = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Epochs:{num_epochs} | Rank:{rank} | Images:{len(images)}')

    pipe = ZImagePipeline.from_pretrained(
        'Tongyi-MAI/Z-Image-Turbo',
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    transformer  = pipe.transformer
    vae          = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer    = pipe.tokenizer
    del pipe; gc.collect()

    print('Encoding images...')
    vae.to(device); text_encoder.to(device)
    all_latents, all_text_embs = [], []
    for i in range(0, len(images), 2):
        img_t = torch.tensor(np.array(images[i:i+2]), dtype=torch.bfloat16).to(device)
        with torch.no_grad():
            lat = vae.encode(img_t).latent_dist.sample()
            all_latents.append((lat * vae.config.scaling_factor).cpu())
            tok = tokenizer(captions[i:i+2], padding='max_length',
                           max_length=256, truncation=True,
                           return_tensors='pt').to(device)
            all_text_embs.append(text_encoder(**tok)[0].cpu())
        del img_t, lat, tok
        torch.cuda.empty_cache()

    vae.cpu(); text_encoder.cpu()
    del vae, text_encoder, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    all_latents   = torch.cat(all_latents, dim=0)
    all_text_embs = torch.cat(all_text_embs, dim=0)
    print(f'Encoded {len(all_latents)} images. Shape: {all_latents.shape}')

    transformer.to(device)
    for p in transformer.parameters():
        p.requires_grad = False

    print(f'Injecting LoRA (rank={rank}) into all attention layers...')
    lora_params = []

    for name, module in transformer.named_modules():
        if not (hasattr(module, 'weight') and isinstance(module, torch.nn.Linear)):
            continue
        if not any(k in name for k in ['to_k', 'to_q', 'to_v', 'to_out']):
            continue

        lora_a = torch.nn.Parameter(
            torch.randn(rank, module.in_features,
                       dtype=torch.bfloat16, device=device) * 0.01)
        lora_b = torch.nn.Parameter(
            torch.zeros(module.out_features, rank,
                       dtype=torch.bfloat16, device=device))
        module.register_parameter('lora_a', lora_a)
        module.register_parameter('lora_b', lora_b)
        lora_params.extend([lora_a, lora_b])

        def make_forward(orig_weight, orig_bias, r=rank):
            def forward(self, x):
                base = torch.nn.functional.linear(x, orig_weight, orig_bias)
                lora = (x @ self.lora_a.T) @ self.lora_b.T * (1.0 / r)
                return base.detach() + lora
            return forward

        module.forward = make_forward(
            module.weight.detach(), 
            module.bias.detach() if module.bias is not None else None
        ).__get__(module, module.__class__)

    print(f'Trainable params: {sum(p.numel() for p in lora_params):,}')
    print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')

    optimizer = AdamW(lora_params, lr=lr, weight_decay=0.01)
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(optimizer, max_lr=lr,
                           total_steps=num_epochs * len(all_latents),
                           pct_start=0.1)

    all_latents   = all_latents.to(device)
    all_text_embs = all_text_embs.to(device)

    def get_pred(out):
        raw = out.sample if hasattr(out, 'sample') else out[0]
        if isinstance(raw, (list, tuple)):
            return torch.cat([t.to(device) for t in raw], dim=0)
        return raw.to(device)

    print(f'\nStarting {num_epochs} epochs...')
    transformer.eval()
    best_loss = float('inf')
    t0 = time.time()
    T  = 1000

    for epoch in range(num_epochs):
        epoch_loss, n = 0, 0
        perm = torch.randperm(len(all_latents))

        for idx in perm:
            lat = all_latents[idx:idx+1]
            emb = all_text_embs[idx:idx+1]
            ts     = torch.randint(0, T, (1,), device=device).long()
            noise  = torch.randn_like(lat)
            t_f    = (ts.float() / T).view(-1, 1, 1, 1)
            noisy  = (1.0 - t_f) * lat + t_f * noise
            target = (noise - lat).unsqueeze(2)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out  = transformer(x=noisy.unsqueeze(2), t=ts, cap_feats=emb)
                pred = get_pred(out).view(target.shape)
                loss = torch.nn.functional.mse_loss(pred.float(), target.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
            epoch_loss += loss.item(); n += 1

        avg  = epoch_loss / n
        ela  = (time.time() - t0) / 60
        eta  = (ela / (epoch+1)) * (num_epochs - epoch - 1)
        print(f'Epoch [{epoch+1:>3}/{num_epochs}]  '
              f'Loss:{avg:.5f}  Elapsed:{ela:.1f}m  ETA:{eta:.1f}m')

        if avg < best_loss:
            best_loss = avg
            os.makedirs(output_dir, exist_ok=True)
            state = {name: {'lora_a': m.lora_a.data.cpu(),
                            'lora_b': m.lora_b.data.cpu()}
                     for name, m in transformer.named_modules()
                     if hasattr(m, 'lora_a')}
            torch.save(state, os.path.join(output_dir, 'best_lora.pt'))
            print(f'  ✓ Best checkpoint saved (loss:{best_loss:.5f})')

        if (epoch+1) % 10 == 0:
            state = {name: {'lora_a': m.lora_a.data.cpu(),
                            'lora_b': m.lora_b.data.cpu()}
                     for name, m in transformer.named_modules()
                     if hasattr(m, 'lora_a')}
            torch.save(state, os.path.join(output_dir, f'lora_epoch_{epoch+1}.pt'))
            print(f'  ✓ Checkpoint: lora_epoch_{epoch+1}.pt')

    state = {name: {'lora_a': m.lora_a.data.cpu(),
                    'lora_b': m.lora_b.data.cpu()}
             for name, m in transformer.named_modules()
             if hasattr(m, 'lora_a')}
    torch.save(state, os.path.join(output_dir, 'final_lora.pt'))

    total_t = (time.time() - t0) / 60
    print(f'\nDone! {num_epochs} epochs in {total_t:.1f} mins | Best loss:{best_loss:.5f}')
    return f'Loss:{best_loss:.5f} | Time:{total_t:.1f}m'


# ── MAIN ─────────────────────────────────────────────────
DATASET_PATH = r'C:\Users\sarja\OneDrive\Desktop\lora_project\dataset\final'
OUTPUT_DIR   = r'C:\Users\sarja\OneDrive\Desktop\lora_project\output'

print('=' * 55)
print('Z-Image LoRA Training | 50 epochs | Full dataset')
print('RTX 4090 | Rank 8 | All attention layers')
print('=' * 55)

print('\nStep 1: Preprocessing on Laptop...')
images, captions = ray.get(preprocess_images.remote(DATASET_PATH))
print(f'Done! {len(images)} images.\n')

print('Step 2: Training on RTX 4090...')
result = ray.get(finetune_lora.remote(
    images, captions, OUTPUT_DIR,
    num_epochs=50, lr=3e-4, rank=8))
print(result)
ray.shutdown()
