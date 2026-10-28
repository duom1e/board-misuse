"""Audit completed fixed trial, render predictions, and record rejection."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;E=B/'kitchen_cutting_trial_v1'
def main():
    r=json.loads((E/'results.json').read_text(encoding='utf-8'));summary={}
    for name,v in r['evaluations'].items():
        positives=[x for x in v['training_fit'] if x['expected'] is not None]
        negatives=[x for x in v['training_fit'] if x['expected'] is None]
        summary[name]=dict(training_positive_matches=sum(x['matched'] for x in positives),training_positive_count=len(positives),training_negative_false_frames=sum(bool(x['predictions']) for x in negatives),training_negative_count=len(negatives),regression=v['regression'])
        sheet=Image.new('RGB',(1280,1900));d=ImageDraw.Draw(sheet)
        for i,x in enumerate(v['training_fit']):
            im=Image.open(E/'images'/f"{x['seconds']}.jpg").convert('RGB');dr=ImageDraw.Draw(im)
            if x['expected']:dr.rectangle(x['expected'],outline='lime',width=2)
            for q in x['predictions']:dr.rectangle(q['box'],outline='orange',width=3)
            a=i%2*640;b=i//2*380;sheet.paste(im,(a,b));d.text((a,b+362),f"{x['seconds']} match={x['matched']} {name}",fill='white')
        sheet.save(E/f'{name}_fit_review.jpg')
    active=B/'active_models.json';summary.update(adopted=False,reason='Old test regression and only consumed single-source training fit',active_models_sha256=hashlib.sha256(active.read_bytes()).hexdigest(),training_performed=True)
    (E/'decision.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (B/'圆木案板训练与回归报告.md').write_text('''# 圆木案板训练与回归报告

2026-10-28完成一次真实YOLO检测器继续训练。旧训练图242张，加新后厨10帧（8个人工案板框、2个无明确可见案板负例），固定10轮，seed42，640输入，batch8。只评固定last.pt，没有挑轮数追分。原默认模型、active_models.json未改动。大模型没有微调。

## 结果

| 检查 | 旧模型 | 本轮模型 |
|---|---:|---:|
| 新训练图8个案板的IoU>=0.5且conf>=0.25匹配 | 0/8 | 5/8 |
| 新训练图2个负例出现候选的帧数 | 1/2 | 0/2 |
| 旧测试24图25目标mAP50 | 0.9117 | 0.8147 |
| 旧测试mAP50–95 | 0.7404 | 0.6213 |
| 旧测试召回 | 0.800 | 0.760 |

新场景有拟合改善，但旧测试显著回退。本轮拒绝替换默认模型，实验权重仅保留供分析。5/8是训练图表现，不能称新后厨泛化准确率；旧测试早已在项目中消费，是回归集而非新盲测。

## 数据与判分修订

300/390秒边缘案板已人工补框，框按画面可见部分，不猜画外区域；6个较清楚圆板框已看原图并叠框核对。420/600秒经原图复核无明确可见案板，之前联系图误标已保留旧文件及修订说明；480秒未知，不参加训练。修订发生于模型输出后，明确不是盲标。没有将身份未确定的棕色食材标成肉类。

旧YOLO在300/330秒虽有输出，但框几乎覆盖整幅，IoU不足0.5，所以本次按定位匹配计0/8；先前2/8仅是候选存在，不能称定位成功。

## 产物与下一步

kitchen_cutting_trial_v1内有locked_plan.json、images/labels、data.yaml、train训练记录/last.pt、results.json、decision.json及新旧叠框图。run_cutting_board_trial.py拒绝重复启动；audit_cutting_board_trial.py离线生成报告和叠框。新视频始终只在训练组，未把相邻帧拆为验证组。素材许可未核实，本批媒体和权重仅作本地实验，不上传。

下一步补不同来源的圆板与边缘/遮挡案板，同时保留旧数据回放；再比较较低学习率或冻结部分层的方案。当前结果不支持加轮数必然变好，也不支持自动判案板错用。身份、接触、生熟和现场用途仍未解决。
''',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
