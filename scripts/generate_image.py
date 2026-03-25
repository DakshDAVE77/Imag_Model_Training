import ray, torch

# Connect to local Ray cluster automatically
try:
    ray.init(address='auto', _temp_dir=r'C:\Users\sarja\AppData\Local\Temp\ray')
except:
    # Fallback for local testing
    ray.init(_temp_dir=r'C:\Users\sarja\AppData\Local\Temp\ray')

# Dynamic Node Discovery
def get_nodes():
    nodes = ray.nodes()
    alive = [n for n in nodes if n.get('Alive')]
    head  = next((n for n in alive if 'node:__internal_head__' in n.get('Resources', {})), alive[0])
    # Laptop is the first worker, or head if solo
    workers = [n for n in alive if n != head]
    laptop  = workers[0] if workers else head
    return head.get('NodeManagerAddress'), laptop.get('NodeManagerAddress')

PC_IP, LAPTOP_IP = get_nodes()
print(f'Connected! PC:{PC_IP} | Laptop:{LAPTOP_IP}')

@ray.remote(num_gpus=1, resources={f'node:{LAPTOP_IP}': 0.01})
def generate(lora_filename, prompt, relative_save_path):
    import torch, os
    from diffusers import ZImagePipeline

    device = torch.device('cuda')
    rank   = 8

    # Dynamically resolve project directory
    home = os.path.expanduser('~')
    if os.path.exists(os.path.join(home, 'OneDrive', 'Desktop', 'lora_project')):
        base_dir = os.path.join(home, 'OneDrive', 'Desktop', 'lora_project')
    elif os.path.exists(os.path.join(home, 'Desktop', 'lora_project')):
        base_dir = os.path.join(home, 'Desktop', 'lora_project')
    else:
        base_dir = r'C:\Users\sarja\OneDrive\Desktop\lora_project'

    lora_path = os.path.join(base_dir, 'output', lora_filename)
    save_path = os.path.join(base_dir, relative_save_path)

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
PROMPT        = 'a photo of a white sks temple, intricate stone carvings, golden spire, dramatic lighting, photorealistic, sharp, 8k'
RELATIVE_SAVE_PATH = r'output\generated\output.png'

print(f'Prompt: {PROMPT}')
result = ray.get(generate.remote(LORA_FILENAME, PROMPT, RELATIVE_SAVE_PATH))
print(result)
ray.shutdown()
