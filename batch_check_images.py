"""Batch image triage using the registered models; never labels a photo compliant."""
import argparse
import copy
import hashlib
import json
import uuid
from collections import Counter
from pathlib import Path

from check_board_misuse import BASE,inspect_image,validate_policy

IMAGE_EXTENSIONS={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}

def write_report(destination,manifest):
    (destination/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    counts=Counter(r['status'] for r in manifest['images'])
    lines=['# 批量图片检查结果','',
        f"运行状态：{manifest['run_status']}；已处理 {len(manifest['images'])}/{manifest['total_images']} 张。",'',
        f"疑似冲突 {counts['suspected_misuse']} 张，待复核 {counts['needs_review']} 张，读取或处理失败 {counts['error']} 张。",'',
        '**以上是程序输出数量，不是违规数量或准确率。没有检测结果也不能判为合规。**','',
        '用途规则由调用者提供，默认视为演示假设。模型不识别生熟、切配动作或清洗历史；厨房案板误检仍较多。','',
        '本命令只做图片推理，不下载数据、不训练、不改变默认模型。逐图结果记录实际权重、阈值和证据。','',
        '| 图片 | 状态 | 案板候选 | 食材框 | 证据/错误 |','|---|---|---:|---:|---|']
    for r in manifest['images']:
        name=r['relative_source'].replace('|','\\|').replace('\n',' ')
        detail=f"[证据图]({r['case_folder']}/evidence.jpg) / [JSON]({r['case_folder']}/result.json)" if r['status']!='error' else r['error'].replace('|','\\|').replace('\n',' ')
        lines.append(f"| {name} | {r['status']} | {r.get('board_candidates','—')} | {r.get('food_detections','—')} | {detail} |")
    (destination/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run_batch(source,config,destination,recursive=False):
    source=Path(source).resolve();destination=Path(destination).resolve()
    validate_policy(config)
    if not source.exists(): raise FileNotFoundError(source)
    if source.is_file():files=[source];root=source.parent
    else:
        root=source
        files=sorted((source.rglob('*') if recursive else source.iterdir()),key=lambda p:str(p).casefold())
        files=[p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.resolve().is_relative_to(destination)]
    if not files:raise ValueError('No supported images found')
    if destination.exists():raise FileExistsError('Output directory already exists; choose a new output path')
    registry_path=BASE/'active_models.json';registry_text=registry_path.read_text(encoding='utf-8')
    registry=json.loads(registry_text)
    spec=validate_policy(config)
    model_keys=(spec['key'],) if config.get('boards') else (config.get('board_model','board'),spec['key'])
    relevant={key:registry[key] for key in model_keys if key in registry}
    provenance={}
    for key,entry in relevant.items():
        weights=Path(entry['weights']);provenance[key]={'weights':str(weights),'sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}
    destination.mkdir(parents=True)
    manifest={'run_status':'running','source':str(source),'total_images':len(files),'policy':copy.deepcopy(config),
        'registry_snapshot':relevant,'weight_provenance':provenance,'images':[],
        'note':'Triage only; no true violation labels or accuracy measurements. Unchanged public photos are not a fresh benchmark.'}
    models={}
    try:
        for index,file in enumerate(files,1):
            if registry_path.read_text(encoding='utf-8')!=registry_text:
                raise RuntimeError('Model registry changed during the run; stopped to avoid mixing model versions')
            relative=file.relative_to(root).as_posix()
            folder=f'{index:04d}_{hashlib.sha256(relative.encode()).hexdigest()[:12]}'
            record={'source':str(file),'relative_source':relative,'case_folder':folder}
            try:
                report=inspect_image(file,copy.deepcopy(config),destination/folder,models=models)
                record.update(status=report['overall_status'],board_candidates=len(report['board_candidates']),food_detections=len(report['food_detections']))
            except Exception as error:
                record.update(status='error',error=f'{type(error).__name__}: {error}')
            manifest['images'].append(record)
            write_report(destination,manifest)
            print(f"{index}/{len(files)} {relative}: {record['status']}",flush=True)
        manifest['run_status']='completed_with_errors' if any(r['status']=='error' for r in manifest['images']) else 'completed'
    except BaseException:
        manifest['run_status']='interrupted_or_failed'
        raise
    finally:write_report(destination,manifest)
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',help='Local image or folder; images are not uploaded')
    policy=parser.add_mutually_exclusive_group(required=True)
    policy.add_argument('--allowed',nargs='+',help='Declare one allowed-food policy for ALL board candidates in ALL images')
    policy.add_argument('--config',help='Policy JSON; a fixed normalized ROI applies identically to every image')
    parser.add_argument('--food-model',default=None,help='Registry food model; new batch entry defaults to foodseg')
    parser.add_argument('--board-model',default=None,help='board is default; board_lvis is experimental opt-in')
    parser.add_argument('--food-confidence',type=float)
    parser.add_argument('--board-confidence',type=float)
    parser.add_argument('--recursive',action='store_true')
    parser.add_argument('--output',default=None,help='New result folder; existing folders are never overwritten')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if args.config else {'all_detected_boards_allowed_foods':args.allowed}
    if args.food_model is not None:config['food_model']=args.food_model
    if args.board_model is not None:config['board_model']=args.board_model
    config.setdefault('food_model','foodseg');config.setdefault('demo_not_site_policy',True)
    for arg,key in [(args.food_confidence,'food_confidence'),(args.board_confidence,'board_confidence')]:
        if arg is not None:config[key]=arg
    destination=Path(args.output) if args.output else BASE/'batch_results'/f'run_{uuid.uuid4().hex[:10]}'
    try:manifest=run_batch(args.source,config,destination,args.recursive)
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as error:parser.error(str(error))
    print(f"Report: {destination.resolve()/'summary.md'}")
    if manifest['run_status']=='completed_with_errors':raise SystemExit(2)

if __name__=='__main__':main()
