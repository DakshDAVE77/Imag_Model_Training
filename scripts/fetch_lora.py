import ray
import os

ray.init(address='192.168.1.31:6379', _temp_dir=r'C:\Users\sarja\AppData\Local\Temp\ray', ignore_reinit_error=True)

@ray.remote(resources={'node:192.168.1.31': 0.01})
def read_lora_from_pc():
    # Try the user path first
    paths = [
        r'C:\Users\sarja\OneDrive\Desktop\lora_project\output\best_sd_lora.pt',
        r'C:\Users\WildMindAi\OneDrive\Desktop\lora_project\output\best_sd_lora.pt',
        r'C:\Users\WildMindAi\Desktop\lora_project\output\best_sd_lora.pt'
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return f.read(), p
    return None, None

print("Fetching file from PC...")
data, found_path = ray.get(read_lora_from_pc.remote())

if data:
    local_path = r'C:\Users\sarja\OneDrive\Desktop\lora_project\output\best_sd_lora.pt'
    with open(local_path, 'wb') as f:
        f.write(data)
    print(f"Success! Fetched from {found_path} and saved to {local_path}")
else:
    print("File not found on PC!")

ray.shutdown()
