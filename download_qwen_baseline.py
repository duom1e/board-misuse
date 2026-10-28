"""Download official Qwen weights to D:, recording exact revision. No inference/training."""
import os, json, traceback
from pathlib import Path
B = Path(__file__).resolve().parent
E = B / 'experiments/qwen3vl_frozen_baseline'
E.mkdir(parents=True, exist_ok=True)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = str(B.parent / 'work/hf-cache')
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
def save(data):
    (E / 'download_status.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
if __name__ == '__main__':
    try:
        import requests
        from huggingface_hub import snapshot_download
        response = requests.get('https://hf-mirror.com/api/models/Qwen/Qwen3-VL-2B-Instruct', timeout=40)
        response.raise_for_status()
        revision = response.json()['sha']
        save({'stage': 'downloading', 'revision': revision, 'model': 'Qwen/Qwen3-VL-2B-Instruct'})
        folder = snapshot_download('Qwen/Qwen3-VL-2B-Instruct', revision=revision,
            local_dir=str(B.parent / 'work/models/Qwen3-VL-2B-Instruct'),
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'README.md', 'LICENSE'], max_workers=3)
        save({'stage': 'downloaded', 'revision': revision, 'folder': folder, 'inference_performed': False})
    except Exception as exc:
        save({'stage': 'failed', 'error': repr(exc)})
        traceback.print_exc()
        raise
