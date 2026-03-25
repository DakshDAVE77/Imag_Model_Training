import ray
import torch

# Connect to local Ray cluster automatically
try:
    ray.init(address='auto', _temp_dir=r'C:\Users\sarja\AppData\Local\Temp\ray')
except:
    # Fallback for local testing if no cluster is running
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
print(f'Connected to cluster! PC:{PC_IP} | Laptop:{LAPTOP_IP}')
print('Available resources:', ray.available_resources())

# Force task onto PC's RTX 4090
@ray.remote(num_gpus=1, resources={f'node:{PC_IP}': 0.01})
def run_on_pc_gpu():
    import torch
    return f'PC GPU ({PC_IP}): {torch.cuda.get_device_name(0)}'

# Force task onto Laptop's GPU (Dynamic IP)
@ray.remote(num_gpus=1, resources={f'node:{LAPTOP_IP}': 0.01})
def run_on_laptop_gpu():
    import torch
    return f'Laptop GPU ({LAPTOP_IP}): {torch.cuda.get_device_name(0)}'

# Run both simultaneously
pc_future     = run_on_pc_gpu.remote()
laptop_future = run_on_laptop_gpu.remote()

results = ray.get([pc_future, laptop_future])
for r in results:
    print(r)

ray.shutdown()
