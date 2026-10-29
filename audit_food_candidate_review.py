"""Export a reviewable, source-bound diagnostic without replacing model outputs."""
import json, hashlib, html
from collections import Counter
from pathlib import Path
B=Path(__file__).resolve().parent
E=B/'food_candidate_review_v1'
def read(name):return json.loads((E/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    result=read('results.json');plan=read('locked_plan.json');refs=read('reference_labels.json')
    assert result['plan_sha256']==refs['plan_sha256']==sha(E/'locked_plan.json')
    rows=result['rows'];assert len(rows)==len(plan['entries'])==len(refs['rows'])
    assert len({r['id'] for r in rows})==len(rows)
    reference={r['id']:r for r in refs['rows']};planned={r['id']:r for r in plan['entries']};summary={};cards=[];queue=[]
    for r in rows:
        ref=reference[r['id']]
        assert r['original_region']==planned[r['id']]['original_region']
        assert sha(r['image'])==r['image_sha256'] and sha(r['crop'])==r['crop_sha256']==ref['crop_sha256']
        assert not r['detection_removed'] and not r['training_eligible'] and not r['physical_contact_verified']
        original=r['original_region'].get('category',r['original_region'].get('location'))
        s=summary.setdefault(r['origin'],dict(total=0,scored=0,reference_consistent=0,unknown_opinions=0,mismatches=[],opinions=Counter(),comparisons=Counter()))
        s['total']+=1;s['opinions'][r['opinion']]+=1;s['comparisons'][r['comparison']]+=1
        s['unknown_opinions']+=r['opinion']=='unknown'
        if ref['score_eligible']:
            s['scored']+=1;s['reference_consistent']+=r['opinion']==ref['reference_category']
            if r['opinion']!=ref['reference_category']:s['mismatches'].append(r['id'])
        needs_attention=(not ref['score_eligible'] or r['comparison']=='different_label' or r['opinion']=='unknown' or (ref['score_eligible'] and r['opinion']!=ref['reference_category']))
        if needs_attention:queue.append(dict(id=r['id'],source=r['source'],seconds=r['seconds'],reference=ref,original=original,opinion=r['opinion'],training_eligible=False))
        note='参考不确定，不计分' if not ref['score_eligible'] else '非专家裁剪参考：'+ref['reference_category']
        cards.append(f'<article><h3>{html.escape(r["id"])}</h3><p>{r["source"]} · {r["seconds"]:.2f}s</p><img src="{Path(r["crop"]).name}"><p>原始：{html.escape(original)}<br>复核意见：{r["opinion"]}<br>{note}</p><details><summary>原图与范围</summary><a href="{Path(r["image"]).as_uri()}">打开原图</a><p>裁剪坐标：{r["crop_xyxy"]}</p><p>位置关系未修改；未确认物理接触。</p></details></article>')
    payload=dict(summary=summary,review_queue=queue,checks=dict(count=len(rows),source_and_crop_hashes_verified=True,original_regions_preserved=True),blind_test=False,training_performed=False)
    (E/'audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    page='<!doctype html><meta charset="utf-8"><title>食材候选逐框复核</title><style>body{font:16px sans-serif;background:#eef2f5;padding:24px;color:#18222c}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}article{background:white;padding:16px;border-radius:10px}img{width:100%;height:230px;object-fit:contain}h1{font-size:26px}</style><h1>食材候选逐框复核 · 43 个区域</h1><p>已消费的开发视频；非盲测、非完整目标标注。参考标签由助手看图整理，尚未经专家确认。大模型意见不覆盖原检测，不判断错用，不直接加入训练。</p><main>'+''.join(cards)+'</main>'
    (E/'index.html').write_text(page,encoding='utf-8')
    lines=['# 食材候选完整复核批次','', '固定上一批 36 帧的全部 27 个 YOLO 食材框，以及 16 个因位置不确定或位于容器而未分类的 VLM 食材框，共 43 个。沿用原分类提示词、10% 边距、512 缩图、12 token、冻结 4B 模型；一次推理，无针对结果修改提示或重跑。','', '先查看全部裁剪并锁定非专家参考，再运行本轮推理。37 个可比较参考、6 个微小或遮挡参考不确定。两处没有明确食材，参考为 unknown；该标签不等于已证实完全没有食物。','', '|候选来源|数量|可比较参考|与参考一致|模型 unknown|','|---|---:|---:|---:|---:|']
    for k,s in summary.items():lines.append(f'|{k}|{s["total"]}|{s["scored"]}|{s["reference_consistent"]}|{s["unknown_opinions"]}|')
    lines+=['','这是重复消费来源上的裁剪类别诊断，不是检测准确率、错用准确率或独立泛化测试。部分裁剪含字幕，部分来源有相邻帧及重复框；裁剪不代表独立事件。','', '## 原始意见和风险','']
    for k,s in summary.items():lines.append(f'- {k}：类别分布 {dict(s["opinions"])}；与原 YOLO 标签比较 {dict(s["comparisons"])}；与可比较参考不同 {s["mismatches"]}。')
    lines+=['','容器或位置不确定候选即便判出肉/菜，仍保留原位置；不能将其重新认定为板上食材。YOLO 与复核不同只是需要复核的意见，不自动纠错或删框。全部候选 training_eligible=false。', '', '## 具体失败与决策', '', '- 00_yolo_00：手和刀占主体，没有清楚食材；原 YOLO vegetable，复核 meat。两者都没有解决误检，不能称纠正。', '- 08_yolo_00：鸡骨架有肉组织；vegetable → meat，与本轮参考一致。', '- 31_yolo_00：手遮水果；meat → vegetable，与本轮参考一致。', '- 23_yolo_00：鸡肉容器局部；meat → mixed，与本轮参考不一致，说明复核也会误伤原正确类别。', '- 32 两个近重复手部框均 unknown；参考也不确定，不计为成功纠错。', '- 16 个位置未过门槛的候选得到 10 meat、5 vegetable、1 unknown；这只是分类诊断，不恢复板上归属。', '', '下一步应保留以上困难区域作为固定开发回归，以不同视频来源扩充手/刀非食材与小食材正例；不能将此次预测直接作为训练标签，也不应自动部署覆盖规则。', '', '## 文件','', '- index.html：43 个区域的逐框对照。','- locked_plan.json / reference_labels.json：推理前候选与参考。','- results.json：原始输出与独立意见。','- audit.json：来源/裁剪 SHA、分组统计和待复核队列。','', '本轮未训练、未切换默认模型。32 项相关软件测试通过；不代表识别准确率。页面仅生成并静态检查，未做浏览器交互验收。']
    (E/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
