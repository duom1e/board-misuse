"""Image-level support labels, explicitly not contact/box/instance annotations."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent
E=B/'food_support_relation_v1'
def main():
    E.mkdir(exist_ok=True)
    assert not (E/'manifest.json').exists(),'Do not overwrite reviewed labels'
    previous=json.loads((B/'experiments/qwen_container_relation_v1/locked_plan.json').read_text(encoding='utf-8'))
    records=[]
    for r in previous['rows']:
        if r['case']=='pepper_container_3':continue # loose tabletop peppers + containers, needs separate multi-region labels
        case=r['case']; positive=case.startswith('consumed_positive')
        source='BV1264y1F7a6' if 'chicken' in case or case.endswith('meat') else 'BV1Ys411L7r6' if positive else 'BV1XN411q7XJ'
        support=['board'] if positive else ['container','hand'] if case=='pepper_container_4' else ['container']
        records.append({'image':r['image'],'sha256':r['sha256'],'source_group':source,'case':case,
            'board_visibility':r['expected_board'],'foreground_food_support':support,
            'visible_food_on_board':True if positive else 'uncertain' if case in ['pepper_container_4','chicken_serving_bowl'] else 'not_applicable',
            'food_category':'meat' if 'chicken' in case or case.endswith('meat') else 'vegetable',
            'board_food_occlusion':'partially_occluded' if case in ['pepper_container_4','chicken_serving_bowl'] else 'not_applicable',
            'role':'consumed_development_candidate','review_status':'pending_overlay_review',
            'annotation_scope':'Dominant visible food group support; no exhaustive instances, boxes, physical-contact proof or hygiene labels.'})
    manifest={'records':records,'review_status':'pending_overlay_review','schema_version':1,
        'excluded':['pepper_container_3: tabletop loose peppers plus jars need separate regional annotations'],
        'split_policy':'All consumed development. No train/val/test split; keep complete source groups together in future splits. Do not claim independent validation.',
        'training_ready':False,'reason':'Tiny source-confounded development labels; no independent validation sources. No automatic compliance targets.'}
    (E/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(1200,5*260),(20,20,20));draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(records):
        im=Image.open(r['image']).convert('RGB');im.thumbnail((590,205));x=i%2*600;y=i//2*260;sheet.paste(im,(x,y))
        draw.text((x+5,y+208),r['case'],fill='white')
        draw.text((x+5,y+225),f"board={r['board_visibility']} support={'+'.join(r['foreground_food_support'])}",fill='white')
        draw.text((x+5,y+242),f"board_food={r['visible_food_on_board']}",fill='white')
    sheet.save(E/'review.jpg')
if __name__=='__main__':main()
