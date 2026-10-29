"""Load one local model for multiple review images; no remote inference."""
import json,os
from pathlib import Path
from evaluate_qwen4b_regions import PROMPT
B=Path(__file__).resolve().parent
def resolve_local_model():
    override=os.environ.get('BOARD_VLM_MODEL_DIR')
    if override:
        folder=Path(override).expanduser().resolve()
        if not (folder/'config.json').is_file():raise ValueError('BOARD_VLM_MODEL_DIR must contain config.json')
        return {'folder':str(folder),'revision':None,'source':'explicit_local_directory_unverified_revision'}
    metadata=B/'experiments/qwen4b_frozen_baseline/download_status.json'
    if not metadata.is_file():raise FileNotFoundError('Local VLM metadata missing. Set BOARD_VLM_MODEL_DIR to the local Qwen3-VL model directory. No download is performed.')
    return json.loads(metadata.read_text(encoding='utf-8'))
class LocalRegionModel:
    def __init__(self):
        import torch
        from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
        meta=resolve_local_model()
        self.revision=meta['revision'];self.torch=torch
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(meta['folder'],dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
        self.processor=AutoProcessor.from_pretrained(meta['folder'],local_files_only=True)
    def predict(self,image,prompt=PROMPT,max_new_tokens=300):
        small=image.convert('RGB').copy();small.thumbnail((512,512))
        messages=[{'role':'user','content':[{'type':'image','image':small},{'type':'text','text':prompt}]}]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(self.model.device)
        with self.torch.inference_mode():out=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
        return self.processor.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
