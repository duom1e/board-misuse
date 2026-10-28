"""Offline audit of locked, completed predictions; never calls a model."""
import csv
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parent
E = B / 'context_new_source_v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan = json.loads((E/'locked_plan.json').read_text(encoding='utf-8'))
    manifest = json.loads((E/'manifest.json').read_text(encoding='utf-8'))
    assert sha(E/'manifest.json') == plan['manifest_sha256']
    result = json.loads((E/'results.json').read_text(encoding='utf-8'))
    expected = {r['case']: r for r in manifest['records']}
    rows = []
    for group in result['rows']:
        for r in group['regions']:
            m = expected[r['case']]
            assert sha(Path(m['source_image'])) == m['source_sha256']
            assert sha(Path(m['image'])) == m['image_sha256']
            assert all(r[k] == m[k] for k in ('expected', 'scored', 'expected_meat_evidence'))
            assert r['category_crop_xyxy'] == m['category_crop_xyxy']
            comparison = r.get('category_context_comparison', {})
            assert not comparison.get('category_replaced', False)
            rows.append(dict(case=r['case'], scored=r['scored'], expected=r['expected'],
                predicted=r['food_category'], exact=r['expected']==r['food_category'],
                meat_target=r['expected_meat_evidence'],
                meat_retained=r['food_category'] in ('meat','mixed'),
                context=group['context_review']['context_category'],
                relation=comparison.get('relation', 'not_run'),
                crop=m['image'], source=m['source_image']))
    assert len(rows)==len(expected) and len({r['case'] for r in rows})==len(expected)
    scored = [r for r in rows if r['scored']]
    meat = [r for r in scored if r['meat_target']]
    audit = dict(manifest_sha256=sha(E/'manifest.json'), results_sha256=sha(E/'results.json'),
        scored_count=len(scored), exact_agreement=sum(r['exact'] for r in scored),
        meat_target_count=len(meat), meat_evidence_retained=sum(r['meat_retained'] for r in meat),
        contradiction_count=sum(r['relation']=='contradiction' for r in rows),
        full_image_calls=sum(g['context_review']['context_inference_performed'] for g in result['rows']),
        training_performed=False, adopted=False, rows=rows)
    (E/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    with (E/'audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 新来源局部与整图复核验证', '',
        '来源：BV1qJ411672x；同一视频的5帧、16个人工裁剪。推理前锁定13个计分样本、3个不确定压力样本；模型、提示词及裁剪参数未调优。', '',
        f"计分类别一致 {audit['exact_agreement']}/{len(scored)}；5个含肉目标中仅2个保留肉类意见，另3个小区域返回unknown。4个蔬菜及4个非食物均符合预标注。", '',
        '整图实际调用4次，1帧局部全未知而跳过；矛盾提示0个。整图未恢复3个小区域肉类证据。unknown是弃答，不能算无肉，也不能自动放行。', '',
        '3个压力样本不计分：模糊木耳unknown、小蒜vegetable、细条meat。细条身份原本就不确定，不能据此确诊错误；但mixed整图允许局部肉或菜，无法用该规则核验局部具体身份。', '',
        '| 样本 | 计分 | 预标注 | 局部输出 | 整图输出 | 关系 |',
        '|---|---|---|---|---|---|']
    lines += [f"| {r['case']} | {r['scored']} | {r['expected']} | {r['predicted']} | {r['context']} | {r['relation']} |" for r in rows]
    lines += ['', '## 判断与下一步', '',
        '上下文功能保留为默认关闭的人工复核辅助，不作为自动纠错或报警依据。本轮没有训练、微调或切换生产模型。后续重点应是有位置标注的小目标数据和未知样本复核，而非反复修改本视频提示词追分。', '',
        '本批数据已经用于诊断，不再作为未来独立测试集。若转入训练，须另留整段来源隔离的测试视频，并先核实素材训练许可；当前公开可见不等于获准再分发或训练。', '',
        '这是人工指定ROI的类别阶段诊断；board_candidate仅为调用该阶段的内部标志，不代表检测到案板或接触。手工非专家标签、同源相关样本、已预训练重叠未知，不能宣称现场准确率或错用识别通过。', '',
        '两个熟肉小裁剪在推理前看图确认邻近木耳，最终锁定标签为mixed；crop_proposals及旧联系图文字保留初稿meat，仅manifest.json是计分依据。没有事后改标签。', '',
        'audit.json/csv为逐项审计，核验manifest及每个原图/裁剪SHA；results.json保留原始意见。']
    (B/'新来源上下文复核验证报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='rows'},ensure_ascii=False))

if __name__=='__main__': main()
