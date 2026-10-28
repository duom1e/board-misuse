"""Load one local model for multiple review images; no remote inference."""
import json
from pathlib import Path
from evaluate_qwen4b_regions import PROMPT
B=Path(__file__).resolve().parent
class LocalRegionModel:
    def __init__(self):
        import torch
        from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
        meta=json.loads((B/'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
        self.revision=meta['revision'];self.torch=torch
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(meta['folder'],dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
        self.processor=AutoProcessor.from_pretrained(meta['folder'],local_files_only=True)
    def predict(self,image,prompt=PROMPT,max_new_tokens=300):
        small=image.convert('RGB').copy();small.thumbnail((512,512))
        messages=[{'role':'user','content':[{'type':'image','image':small},{'type':'text','text':prompt}]}]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(self.model.device)
        with self.torch.inference_mode():out=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
        return self.processor.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
