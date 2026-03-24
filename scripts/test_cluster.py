import ray
import torch

ray.init(address='192.168.1.41:6379', _temp_dir=r'C:\Users\sarja\AppData\Local\Temp\ray')

print('Connected to cluster!')
print('Available resources:', ray.available_resources())

# Force task onto PC's RTX 4090
@ray.remote(num_gpus=1, resources={'node:192.168.1.41': 0.01})
def run_on_pc_gpu():
    import torch
    return f'PC GPU: {torch.cuda.get_device_name(0)}'

# Force task onto Laptop's GPU
@ray.remote(num_gpus=1, resources={'node:192.168.1.17': 0.01})
def run_on_laptop_gpu():
    import torch
    return f'Laptop GPU: {torch.cuda.get_device_name(0)}'

# Run both simultaneously
pc_future     = run_on_pc_gpu.remote()
laptop_future = run_on_laptop_gpu.remote()

results = ray.get([pc_future, laptop_future])
for r in results:
    print(r)

ray.shutdown()
