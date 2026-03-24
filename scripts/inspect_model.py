import ray

ray.init(address='192.168.1.41:6379', _temp_dir=r'C:\Users\sarja\AppData\Local\Temp\ray')

@ray.remote(num_gpus=1, resources={'node:192.168.1.41': 0.01})
def inspect_model():
    import inspect
    from diffusers import ZImagePipeline
    import torch

    pipe = ZImagePipeline.from_pretrained(
        'Tongyi-MAI/Z-Image-Turbo',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False
    )

    # Print the exact forward signature
    sig = inspect.signature(pipe.transformer.forward)
    print("\n=== ZImageTransformer2DModel.forward() signature ===")
    for name, param in sig.parameters.items():
        print(f"  {name}: default={param.default}")

    # Also print the source file location
    print("\n=== Source file ===")
    print(inspect.getfile(pipe.transformer.__class__))

    return "Done"

result = ray.get(inspect_model.remote())
print(result)
ray.shutdown()
