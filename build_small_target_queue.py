"""Prepare traceable review candidates, not automatic training labels."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

B=Path(__file__).resolve().parent
E=B/'small_target_review_v1'

def main():
    E.mkdir(exist_ok=True)
    source=B/'context_new_source_v1/audit.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    queue=[]
    for r in data['rows']:
        if (r['scored'] and not r['exact']) or not r['scored']:
            queue.append(dict(r, review_state='pending_expert', training_eligible=False,
                split='consumed_development_only', license_state='unverified',
                reason='small_target_abstention' if r['scored'] else 'predeclared_ambiguous',
                required_review=['identity','tight_instance_box','occlusion','visible_contact_unknown','source_permission']))
    output=dict(source_audit_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        records=queue, training_performed=False, automatic_labels_accepted=False)
    (E/'queue.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    canvas=Image.new('RGB',(900,300*len(queue)), '#202020')
    draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(queue):
        im=Image.open(r['crop']).convert('RGB');im.thumbnail((420,250))
        canvas.paste(im,(0,300*i))
        draw.text((440,300*i+20),f"{r['case']}\nexpected: {r['expected']}\nmodel: {r['predicted']}\n{r['reason']}\nNOT approved for training",fill='white')
    canvas.save(E/'review.jpg')
    (E/'README.md').write_text('''# 小目标复核队列

6个候选来自已经消费的开发视频：3个计分小区域弃答，3个推理前声明的不确定样本。queue.json保存原图、裁剪、预标注、原模型意见与来源审计SHA；没有生成新的训练真值。

复核时需确认食材身份、标紧实例框、记录遮挡，并保留无法确定。混合区域不能凭肉纹理就整框标肉；案板接触不能凭二维重叠确认。全部默认pending_expert、training_eligible=false，素材许可未核实。

后续训练数据应补充不同拍摄来源的小肉块、小蔬菜和刀具/木纹等非食物；同视频相邻帧及同图裁剪必须留在同一分组。本队列不可再当独立测试集。另留未参与选样和调参的完整来源测试，避免随机拆帧造成泄漏。

review.jpg仅放大展示原裁剪，不做生成、修复或补细节。该队列用于补标准备，不代表已完成检测数据标注，也没有启动训练。
''',encoding='utf-8')
    print(f'Prepared {len(queue)} review candidates; training eligible: 0')

if __name__=='__main__':main()
