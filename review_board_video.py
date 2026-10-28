"""Sampled short-video review. No tracking, temporal filling, compliance or realtime claim."""
import argparse,json,hashlib,csv,time
from pathlib import Path
from video_sample_plan import plan_frames
from region_response_validation import validate_region_response
from food_region_gate import locate
from evaluate_qwen4b_regions import PROMPT
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float,required=True)
    p.add_argument('--interval',type=float,default=2);p.add_argument('--replay',type=Path,help='JSON with video_sha256 and responses keyed by frame index; no VLM inference')
    p.add_argument('--food-categories',action='store_true',help='Also classify board-candidate food crops; live mode only, opinions remain needs_review')
    p.add_argument('--food-context',action='store_true',help='Add a separate full-image category opinion and contradiction hints; requires --food-categories')
    args=p.parse_args()
    if not args.video.is_file():p.error('Video does not exist')
    if args.output_dir.exists():p.error('Output directory must be new')
    if args.replay and args.food_categories:p.error('--food-categories cannot be combined with replay: no saved category responses')
    if args.food_context and not args.food_categories:p.error('--food-context requires --food-categories')
    import cv2
    from PIL import Image,ImageDraw
    cap=cv2.VideoCapture(str(args.video));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:schedule=plan_frames(args.start,args.end,args.interval,fps,count)
    except ValueError as exc:cap.release();p.error(str(exc))
    digest=sha(args.video);replay=None
    if args.replay:
        replay=json.loads(args.replay.read_text(encoding='utf-8'))
        if replay.get('video_sha256')!=digest:p.error('Replay does not match input video SHA256')
        if any(str(r['frame_index']) not in replay.get('responses',{}) for r in schedule):p.error('Replay is missing scheduled frames')
    out=args.output_dir;out.mkdir(parents=True);frames=out/'frames';frames.mkdir()
    def save(name,data):
        temporary=out/(name+'.tmp')
        temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(out/name)
    plan={'video':str(args.video.resolve()),'video_sha256':digest,'fps':fps,'frame_count':count,'schedule':schedule,'prompt':PROMPT,'thumbnail':512,'max_new_tokens':300,'do_sample':False,'mode':'replay' if replay else 'live','scope':'Uniform sampled review only. No tracking, smoothing, event duration, full-fps or compliance judgement. No classification-head filtering.'}
    from food_category_review import FOOD_CATEGORY_PROMPT,review_board_foods
    plan.update(food_category_review=args.food_categories,food_category_prompt=FOOD_CATEGORY_PROMPT if args.food_categories else None,category_max_new_tokens=12 if args.food_categories else None,category_crop_margin=.1,category_min_crop_side_pixels=20)
    from food_context_review import CONTEXT_PROMPT,review_food_context
    plan.update(food_context_review=args.food_context,food_context_prompt=CONTEXT_PROMPT if args.food_context else None,context_max_new_tokens=12 if args.food_context else None)
    if args.replay:plan.update(replay_file=str(args.replay.resolve()),replay_file_sha256=sha(args.replay))
    save('locked_plan.json',plan);save('status.json',{'stage':'loading','mode':plan['mode']})
    runtime=None
    if not replay:
        try:
            from region_vlm_runtime import LocalRegionModel
            runtime=LocalRegionModel();plan['model_revision']=runtime.revision;save('locked_plan.json',plan)
        except Exception as exc:save('status.json',{'stage':'failed','error':repr(exc)});raise
    rows=[]
    for i,sample in enumerate(schedule):
        started=time.perf_counter();row={**sample,'status':'needs_review','compliance_determined':False,'board_state':'unknown','inference_performed':False};stem=f"{sample['frame_index']:08d}"
        cap.set(cv2.CAP_PROP_POS_FRAMES,sample['frame_index']);ok,frame=cap.read()
        if not ok:row['frame_error']={'code':'decode_error','message':'Frame could not be decoded'}
        else:
            path=frames/(stem+'.jpg');cv2.imwrite(str(path),frame);row.update(image=str(path.resolve()),image_sha256=sha(path),decoded_frame_index=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1)
            try:
                if row['decoded_frame_index']!=sample['frame_index']:raise ValueError('Decoded frame index does not match the scheduled frame')
                if replay:
                    record=replay['responses'][str(sample['frame_index'])]
                    if record['image_sha256']!=row['image_sha256']:raise ValueError('Replay image SHA mismatch; response was not applied')
                    answer=record['raw_response']
                else:
                    row['inference_performed']=True;answer=runtime.predict(Image.open(path))
                if not isinstance(answer,str):raise ValueError('Response is not text')
                validation=validate_region_response(answer);row.update(raw_response=answer,output_validation=validation,board_state=validation['board_state'])
                (frames/(stem+'_raw.txt')).write_text(answer,encoding='utf-8')
                if validation['valid']:
                    image=Image.open(path).convert('RGB');w,h=image.size
                    boxes={k:[[a*w/1000,b*h/1000,c*w/1000,d*h/1000] for a,b,c,d in values] for k,values in validation['normalized_boxes'].items()};row['pixel_boxes']=boxes
                    row['food_regions']=[{'box':box,**locate(box,boxes['boards'],boxes['containers'])} for box in boxes['food']]
                    if args.food_categories:row['food_regions']=review_board_foods(image,row['food_regions'],lambda crop:runtime.predict(crop,prompt=FOOD_CATEGORY_PROMPT,max_new_tokens=12))
                    if args.food_context:row['food_regions'],row['food_context_review']=review_food_context(image,row['food_regions'],lambda im:runtime.predict(im,prompt=CONTEXT_PROMPT,max_new_tokens=12))
                    draw=ImageDraw.Draw(image)
                    for key,color in [('boards','cyan'),('containers','yellow'),('food','red')]:
                        for box in boxes[key]:draw.rectangle(box,outline=color,width=4)
                    review=frames/(stem+'_review.jpg');image.save(review);row['review_image']=str(review.resolve())
            except Exception as exc:
                row['frame_error']={'code':'replay_error' if replay else 'inference_or_processing_error','message':repr(exc)};row['board_state']='unknown'
                for key in ['pixel_boxes','food_regions','review_image']:row.pop(key,None)
        row['processing_seconds']=time.perf_counter()-started;rows.append(row)
        save('partial_results.json',{'rows':rows});save('status.json',{'stage':'running','completed_samples':len(rows),'total_samples':len(schedule),'mode':plan['mode']})
        print(f"{i+1}/{len(schedule)} {sample['frame_seconds']:.3f}s {row['board_state']}",flush=True)
    cap.release()
    summary={state:sum(r['board_state']==state for r in rows) for state in ['candidate_present','no_board_detected','unknown']}
    save('results.json',{'plan':plan,'rows':rows,'summary':summary,'status':'needs_review','compliance_determined':False})
    with (out/'timeline.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['frame_index','frame_seconds','board_state','inference_performed','food_categories','context_category','contradictions','error']);writer.writeheader()
        for r in rows:writer.writerow({k:r[k] for k in ['frame_index','frame_seconds','board_state','inference_performed']}|{'food_categories':', '.join(z['food_category'] for z in r.get('food_regions',[]) if z.get('category_inference_performed')),'context_category':r.get('food_context_review',{}).get('context_category',''),'contradictions':r.get('food_context_review',{}).get('contradiction_count',''),'error':'; '.join(e['path']+': '+e['code'] for e in r.get('output_validation',{}).get('errors',[])) or str(r.get('frame_error',''))})
    report=['# 视频采样复核','',f"处理方式：{plan['mode']}；采样数：{len(rows)}。",'',f"案板候选 {summary['candidate_present']}，未检出 {summary['no_board_detected']}，未知 {summary['unknown']}。",'','全部需人工复核。未检出不代表无板、空板或合规。没有跟踪、平滑或沿用前帧，未验证采样点之间的画面。','', '| 视频时间/秒 | 状态 | 问题 |','| --- | --- | --- |']
    for r in rows:
        issue='; '.join(e['path']+': '+e['code'] for e in r.get('output_validation',{}).get('errors',[])) or r.get('frame_error',{}).get('code','')
        report.append(f"| {r['frame_seconds']:.3f} | {r['board_state']} | {issue} |")
    from render_video_review import render_contact_sheets
    pages=render_contact_sheets(out,rows)
    report+=['','## 复核联系图','','红色外边框表示该采样为未知/失败，不能解读为模型定位框。']
    report += [f'- [{page.name}](<{page.resolve().as_posix()}>)' for page in pages]
    if args.food_categories:
        report+=['','## 板上食材候选的类别意见','','类别仅供复核，不能判断生熟、物理接触或错用。容器内/不明确位置不做此步骤；小于20像素的候选保留未知。']
        for r in rows:
            details=[f"{z.get('food_category','unknown')} ({z.get('category_review_status','not_requested')})" for z in r.get('food_regions',[]) if z['location']=='board_candidate']
            report.append(f"- {r['frame_seconds']:.3f}s："+('、'.join(details) if details else '没有可供分类的板上食材候选，不能据此认定空板。'))
    if args.food_context:
        report+=['','## 局部与整图类别的矛盾提示','','两份意见来自同一模型的不同提问，不是独立模型证据。不矛盾也不证明局部为食物、在板上或合规；局部类别始终保留，不被整图覆盖。']
        for r in rows:
            context=r.get('food_context_review',{});report.append(f"- {r['frame_seconds']:.3f}s：整图 {context.get('context_category','未调用')}；类别矛盾 {context.get('contradiction_count',0)} 处。")
    (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    save('status.json',{'stage':'completed','mode':plan['mode'],'summary':summary,'training_performed':False});print(str((out/'report.md').resolve()))
if __name__=='__main__':main()
