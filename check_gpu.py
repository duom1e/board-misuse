import json
import torch

def main():
    info = {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'cuda_available': torch.cuda.is_available()}
    if not info['cuda_available']:
        raise RuntimeError(f'GPU unavailable: {info}')
    info['gpu'] = torch.cuda.get_device_name(0)
    info['vram_gib'] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    model = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, 3)).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    for _ in range(3):
        x = torch.randn(16, 128, device='cuda')
        y = torch.randint(0, 3, (16,), device='cuda')
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    info['gpu_forward_backward_optimizer'] = 'passed'
    print(json.dumps(info, indent=2))

if __name__ == '__main__':
    main()
