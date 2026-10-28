"""Experimental local 4B localization review, with no compliance judgement."""
import argparse,json,hashlib
from pathlib import Path
from food_region_gate import locate
from evaluate_qwen4b_regions import PROMPT
from region_response_validation import validate_region_response
B=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--board-head-review',action='store_true',help='Add experimental board/table head opinions; preserve all original boxes')
    parser.add_argument('--response-file',type=Path,help='Replay a saved raw response for diagnostics; do not load or run the VLM')
    args=parser.parse_args()
    if not args.image.is_file():parser.error('Input image does not exist')
    if args.output_dir.exists():parser.error('Use a new output directory; existing results are preserved')
    from PIL import Image,ImageDraw
    image=Image.open(args.image).convert('RGB');small=image.copy();small.thumbnail((512,512))
    revision=None
    if args.response_file:
        if not args.response_file.is_file():parser.error('Response file does not exist')
        answer=args.response_file.read_text(encoding='utf-8')
    else:
        import torch
        from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
        download=json.loads((B/'experiments/qwen4b_frozen_baseline/download_status.json').read_text(encoding='utf-8'))
        revision=download['revision']
        model=Qwen3VLForConditionalGeneration.from_pretrained(download['folder'],dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
        processor=AutoProcessor.from_pretrained(download['folder'],local_files_only=True)
        messages=[{'role':'user','content':[{'type':'image','image':small},{'type':'text','text':PROMPT}]}]
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
        with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=300,do_sample=False)
        answer=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
    validation=validate_region_response(answer)
    result={'status':'needs_review','compliance_determined':False,'image':str(args.image.resolve()),'sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),'model_revision':revision,'prompt':PROMPT,'raw_response':answer,'inference_performed':not bool(args.response_file),'output_validation':validation,'board_state':validation['board_state']}
    if args.response_file:
        result['response_file']=str(args.response_file.resolve())
        result['response_file_sha256']=hashlib.sha256(args.response_file.read_bytes()).hexdigest()
    if validation['valid']:
        parsed=validation['normalized_boxes']
        w,h=image.size;boxes={k:[[a*w/1000,b*h/1000,c*w/1000,d*h/1000] for a,b,c,d in v] for k,v in parsed.items()}
        result['pixel_boxes']=boxes
        result['food_regions']=[{'box':box,**locate(box,boxes['boards'],boxes['containers'])} for box in boxes['food']]
        result['limitations']='Coarse bounding boxes can include hands, merge objects or miss food. Empty detections do not imply clean, empty board, or compliance.'
        draw=ImageDraw.Draw(image)
        for k,color in [('boards','cyan'),('containers','yellow'),('food','red')]:
            for box in boxes[k]:draw.rectangle(box,outline=color,width=4);draw.text(box[:2],k,fill=color)
    else:result['parse_error']='; '.join(e['path']+': '+e['code'] for e in validation['errors'])
    if args.board_head_review and 'pixel_boxes' in result:
        if not args.response_file:
            del model,inputs,generated
            torch.cuda.empty_cache()
        from board_candidate_review import review_candidates
        try:
            clean_image=Image.open(args.image).convert('RGB')
            result['board_candidate_review']=review_candidates(clean_image,result['pixel_boxes']['boards'])
            result['board_review_scope']='Experimental, uncalibrated opinions only; all original boxes and needs_review preserved.'
        except Exception as exc:result['board_review_error']=repr(exc)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output_dir/'raw_response.txt').write_text(answer,encoding='utf-8')
    if 'parse_error' not in result:image.save(args.output_dir/'review.jpg')
    state_label={'unknown':'未知：模型输出未通过校验','candidate_present':'存在案板候选，需人工复核','no_board_detected':'未检测到案板，不代表画面无板'}[result['board_state']]
    report=['# 图片复核结果','',state_label,'','状态：needs_review，未判定合规。',
            '','处理方式：'+('保存响应回放，未执行本次VLM推理。' if args.response_file else '本次实际执行本地VLM推理。'),
            '','原始响应保存在raw_response.txt，详细结果保存在result.json。']
    if validation['errors']:
        report+=['','## 输出错误','']
        report += [f"- `{e['path']}`：`{e['code']}` — {e['message']}" for e in validation['errors']]
        report+=['','整条结果保持未知，不展开嵌套坐标、不发布部分字段为成功候选，也不生成识别叠框图。']
    else:report+=['','叠框图为review.jpg。格式通过只表示结构有效，不能证明框正确、食材接触或案板使用合规。']
    (args.output_dir/'validation_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(str((args.output_dir/'result.json').resolve()))
if __name__=='__main__':main()
