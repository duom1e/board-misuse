"""Run the experimental location filter on supplied boxes; no model predictions implied."""
import argparse,json
from pathlib import Path
from food_region_gate import locate
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='JSON with food: list of boxes; boards/containers: lists of boxes')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve():parser.error('Output must differ from input')
    if args.output.exists():parser.error('Refusing to overwrite existing output')
    data=json.loads(args.input.read_text(encoding='utf-8'))
    foods=data['food'];boards=data.get('boards',[]);containers=data.get('containers',[])
    if not isinstance(foods,list):parser.error('food must be a list of boxes')
    result={'status':'needs_review','scope':'Coarse supplied-box candidates, not automatic contact or compliance judgement',
        'food_regions':[{'box':box,**locate(box,boards,containers)} for box in foods]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
