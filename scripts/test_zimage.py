import ray
import torch

ray.init(address='192.168.1.11:6379', _temp_dir=r'C:\Users\sarja\AppData\Local\Temp\ray')

@ray.remote(num_gpus=1, resources={'node:192.168.1.11': 0.01})
def test_generation():
    import torch
    from diffusers import ZImagePipeline

    print(f'Running on: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    pipe = ZImagePipeline.from_pretrained(
        'Tongyi-MAI/Z-Image-Turbo',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to('cuda')

    image = pipe(
        prompt='A beautiful mountain landscape at sunset, photorealistic, 4k',
        height=1024, width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=torch.Generator('cuda').manual_seed(42),
    ).images[0]

    image.save(r'C:\Users\sarja\OneDrive\Desktop\lora_project\test_output.png')
    return 'Test generation complete! Image saved.'

result = ray.get(test_generation.remote())
print(result)
ray.shutdown()
