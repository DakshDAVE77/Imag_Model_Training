import ray, torch

ray.init(address='192.168.1.41:6379', _temp_dir=r'C:\Users\sarja\AppData\Local\Temp\ray')

@ray.remote(num_gpus=1, resources={'node:192.168.1.17': 0.01})
def generate(lora_filename, prompt, save_path):
    import torch, os
    from diffusers import ZImagePipeline

    device = torch.device('cuda')
    rank   = 8

    # LoRA is already local on the laptop — saved here by finetune_zimage.py
    lora_path = r'C:\Users\sarja\OneDrive\Desktop\lora_project\output' + '\\' + lora_filename
    print(f'Loading LoRA from {lora_path}...')

    # ── Load model ────────────────────────────────────────
    print('Loading Z-Image-Turbo...')
    pipe = ZImagePipeline.from_pretrained(
        'Tongyi-MAI/Z-Image-Turbo',
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
    pipe.to(device)
    transformer = pipe.transformer

    # ── Inject LoRA weights ───────────────────────────────
    state    = torch.load(lora_path, map_location=device, weights_only=True)
    injected = 0

    for name, module in transformer.named_modules():
        if name not in state:
            continue
        lora_a = state[name]['lora_a'].to(device).to(torch.bfloat16)
        lora_b = state[name]['lora_b'].to(device).to(torch.bfloat16)
        bias   = module.bias

        def make_forward(w, b, la, lb, r=rank):
            def forward(self, x):
                return (torch.nn.functional.linear(x, w, b) +
                        (x @ la.T) @ lb.T * (1.0 / r))
            return forward

        module.forward = make_forward(
            module.weight, bias, lora_a, lora_b).__get__(module, module.__class__)
        injected += 1

    print(f'Injected into {injected} layers')

    # ── Generate ──────────────────────────────────────────
    print(f'Generating: "{prompt}"')
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            height=1024, width=1024,
            num_inference_steps=25,
            guidance_scale=3.5,
            generator=torch.Generator('cuda').manual_seed(42),
        ).images[0]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    print(f'Saved to {save_path}')
    return f'Done! Saved to {save_path}'


# ── MAIN ─────────────────────────────────────────────────
LORA_FILENAME = 'best_lora.pt'
PROMPT        = 'a photo of sks temple, intricate stone carvings, golden spire, dramatic lighting, photorealistic, sharp, 8k'
SAVE_PATH     = r'C:\Users\sarja\OneDrive\Desktop\lora_project\output\generated\output.png'

print(f'Prompt: {PROMPT}')
result = ray.get(generate.remote(LORA_FILENAME, PROMPT, SAVE_PATH))
print(result)
ray.shutdown()
