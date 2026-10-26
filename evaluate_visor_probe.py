"""Frozen model comparison on the new-source VISOR positive-only probe."""
import json,math,os
from pathlib import Path
from PIL import Image,ImageDraw
BASE=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(BASE.parent/'work/yolo-config');os.environ['YOLO_AUTOINSTALL']='False';os.environ['YOLO_OFFLINE']='True'
from run_experiment import box_iou,precision_recall,write_json
from experiment_data import read_boxes,sha256
from ultralytics import YOLO
DATA=BASE/'visor_board_probe';OUT=BASE/'experiments/visor_frozen_probe'

def main():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    plan=json.loads((DATA/'locked_sampling_plan.json').read_text(encoding='utf-8'))
    if len(manifest['records'])<8:raise RuntimeError('Fewer than8 distinct participant frames; insufficient for planned probe')
    OUT.mkdir(parents=True,exist_ok=False)
    locked={'sampling_plan_sha256':sha256(DATA/'locked_sampling_plan.json'),'manifest_sha256':sha256(DATA/'manifest.json'),'models':plan['model_sha256'],'confidence':.5,'iou':.5,'imgsz':640,'test_scope':'new-source positive board localization only','no_promotion_or_tuning':True}
    write_json(OUT/'locked_evaluation.json',locked)
    outputs={}
    for key,weights in plan['models'].items():
        if sha256(weights)!=plan['model_sha256'][key]:raise RuntimeError('Weights changed since sampling plan')
        model=YOLO(weights)
        if list(model.names.values())!=['cutting_board']:raise RuntimeError('Not a cutting-board model')
        rows=[];counts={'tp':0,'fp':0,'fn':0}
        sheet=Image.new('RGB',(1280,((len(manifest['records'])+3)//4)*250),'white')
        for index,r in enumerate(manifest['records']):
            if sha256(r['image'])!=r['image_sha256'] or sha256(r['label'])!=r['label_sha256']:raise RuntimeError('Probe image/label changed')
            prediction=model.predict(r['image'],conf=.5,imgsz=640,device=0,verbose=False)[0]
            h,w=prediction.orig_shape
            truths=[[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h] for _,x,y,bw,bh in read_boxes(Path(r['label']),['cutting_board'])]
            detections=sorted(zip(prediction.boxes.xyxy.cpu().tolist(),prediction.boxes.conf.cpu().tolist()),key=lambda x:-x[1])
            matched=set();details=[]
            for box,confidence in detections:
                overlap,i=max(((box_iou(box,g),i) for i,g in enumerate(truths) if i not in matched),default=(0,None))
                ok=overlap>=.5 and i is not None
                if ok:matched.add(i)
                counts['tp' if ok else 'fp']+=1
                details.append({'box':box,'confidence':confidence,'matched':ok,'gt_index':i if ok else None,'iou':overlap})
            counts['fn']+=len(truths)-len(matched)
            rows.append({'participant':r['participant'],'video':r['video'],'image':r['image'],'truths':truths,'predictions':details,'tp':len(matched),'fp':len(detections)-len(matched),'fn':len(truths)-len(matched)})
            im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
            for g in truths:draw.rectangle(g,outline='orange',width=6)
            for det in details:draw.rectangle(det['box'],outline='cyan',width=4)
            im.thumbnail((315,215));x=index%4*320;y=index//4*250;sheet.paste(im,(x,y));draw=ImageDraw.Draw(sheet)
            draw.text((x+2,y+217),r['video'],fill='black');draw.text((x+2,y+233),f'TP {len(matched)} FP {len(detections)-len(matched)} FN {len(truths)-len(matched)}',fill='black')
        sheet.save(OUT/(key+'_predictions.jpg'))
        outputs[key]={'fixed':precision_recall(counts),'images':len(rows),'participant_frames_with_all_boards_found':sum(row['fn']==0 for row in rows),'rows':rows}
        write_json(OUT/(key+'.json'),outputs[key])
    if sha256(DATA/'manifest.json')!=locked['manifest_sha256']:raise RuntimeError('Probe manifest changed')
    summary={'scope':'New to project, different source, positive-only sparse egocentric frames; not site misuse accuracy.',
             'images':manifest['images'],'boards':manifest['boards'],'participants':manifest['participant_groups'],
             'results':{key:{k:v for k,v in entry.items() if k!='rows'} for key,entry in outputs.items()},
             'limitations':manifest['limitations'],'registry_changed':False,'trained_on_probe':False}
    write_json(OUT/'summary.json',summary)
    lines=['# 第二十二轮识别实验：VISOR新来源冻结评估','','冻结上一轮候选，不训练、不改阈值、不改默认模型。评估只覆盖案板定位，不是用途违规识别。', '',
           f'样本{manifest["images"]}张、{manifest["boards"]}个可见轮廓外接框，来自{manifest["participant_groups"]}个不同参与者。官方val元数据每参与者最多查看前三个视频，选首个案板class18且exhaustive=y的帧；按标注取样，未按模型效果筛图。只选有案板的帧，因此没有负场景误报率。', '',
           '|模型|TP|FP|FN|标注匹配精确率|召回率|F1|','|---|---:|---:|---:|---:|---:|---:|']
    for key,entry in outputs.items():
        m=entry['fixed'];lines.append(f'|{key}|{m["tp"]}|{m["fp"]}|{m["fn"]}|{m["precision"]:.4f}|{m["recall"]:.4f}|{m["f1"]:.4f}|')
    lines+=['','## 解释边界','','RGB来自官方分帧ZIP，通过HTTP Range只下载选定帧；像素标注使用官方稀疏1920x1080坐标系，再归一化映射到图片。选样冻结后保存图像/标签校验值，并与旧图像字节/dHash筛查。每参与者一帧减少相邻帧重复，不证明所有家庭或预训练独立。', '',
            '使用可见mask外接框，不是遮挡部分补全的完整物体框；与旧框标注口径有差别。数量少、元数据顺序选择、仅正图、第一视角均限制推广。即使候选有提升，也不能因此称真实后厨监控可靠或直接替换默认。', '',
            '官方许可CC BY-NC4.0，仅本地非商业研究探测，不发布图片或新权重。作者：Ahmad Darkhalil、Dandan Shan、Bin Zhu、Jian Ma、Amlan Kar、Richard Higgins、Sanja Fidler、David Fouhey、Dima Damen。来源：https://epic-kitchens.github.io/VISOR/site 及 https://doi.org/10.5523/bris.2v6cgv1x04ol22qp9rm9x2j6a7 。转换：仅案板轮廓外接框与证据叠图。', '',
            '评估完成后，本批样本属于已查看评估资料；后续如用来开发，不能再次冒充完全未看过的测试。']
    (BASE/'第二十二轮识别实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
