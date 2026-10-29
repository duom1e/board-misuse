"""Compare on identical old held crops; keep the added source separate."""
import json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent;E=B/'food_presence_head_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 old=read(B/'food_presence_head_v1/results.json');new=read(E/'results.json');plan=read(E/'locked_plan.json');m=B/'food_presence_patches_v3/reviewed_manifest.json';refs=read(m)['records'];byid={r['id']:r for r in refs}
 assert sha(m)==plan['manifest_sha256'] and sha(B/'active_models.json')==plan['active_models_sha256']
 a={r['id']:r for f in old['folds'] for r in f['rows']};b={r['id']:r for f in new['folds'] for r in f['rows']}
 assert len(b)==42 and len(a)==36 and set(a)<=set(b)
 for r in refs:assert sha(r['image'])==r['image_sha256'] and sha(r['crop'])==r['crop_sha256']
 for f in plan['folds']:
  for key in ['source_video_sha256','image_sha256','crop_sha256']:
   assert not ({byid[k][key] for k in f['train_ids']}&{byid[k][key] for k in f['held_ids']})
 changed=[dict(id=k,reference=a[k]['reference'],before=a[k]['prediction'],after=b[k]['prediction']) for k in a if a[k]['prediction']!=b[k]['prediction']]
 stats=dict(old_shared_matches=sum(r['prediction']==r['reference'] for r in a.values()),new_shared_matches=sum(b[k]['prediction']==a[k]['reference'] for k in a),shared_count=36,added_source_matches=sum(r['prediction']==r['reference'] for k,r in b.items() if k not in a),added_count=6,changes=changed,adopted=False,default_unchanged=True,source_frame_crop_overlap=False)
 (E/'comparison.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# 补充另一来源辣椒后的训练对照','','新增另一个已有视频的6个裁剪，逐张看图核对：3个植物食材、3个案板/台面背景。总计5视频42裁剪。仍为非专家开发数据、历史已消费来源。','','沿用上一轮冻结YOLO11s特征、rect=False/640、Adam .01、固定120轮/seed20261029，整视频留出，共实际训练5个分类层。没有微调大模型或检测主干。','','|留出视频|符合参考|','|---|---:|']
 for f in new['folds']:lines.append(f'|{f["held_source"]}|{f["held_matches"]}/{f["held_count"]}|')
 lines+=['',f'同一旧36裁剪对比：{stats["old_shared_matches"]}/36 → {stats["new_shared_matches"]}/36。新增视频单列 {stats["added_source_matches"]}/6，不能用新增简单样本抬高旧集成绩。', '', '原辣椒来源留出从6/10到7/10，但原肉末视频从9/10到8/10，鸡肉6/8和猪肉7/8不变。数据扩充没有带来旧开发集净改善，继续拒绝自动采用。', '', '逐样本类别变化：']
 lines += [f'- {r["id"]}：参考 {r["reference"]}，{r["before"]} → {r["after"]}。' for r in changed]
 lines+=['','来源视频、原帧、裁剪 SHA 跨折无重合核验通过；active_models.json 未变。旧来源均已消费，因此整源留出仍是开发回归，不能视为盲测。','', '本次说明仅增加少量辣椒样本还不足以修复跨来源混淆。下一步需扩充多来源植物形态与皮脂、小肉，并加入遮挡/检测候选宽框；不继续对这42个样本调参追分。','', '训练已全部结束。代码、文字报告可上传；视频、裁剪、权重留本机。']
 (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(stats,ensure_ascii=False))
if __name__=='__main__':main()
