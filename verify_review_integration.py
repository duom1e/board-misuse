import json
from pathlib import Path
from annotation_io import prepare_image
from check_board_misuse import inspect_image
base=Path(__file__).resolve().parent
source=base/'board_dataset/images/test/37d8f468504ad26c.jpg'
folder, image, metadata=prepare_image(source,base/'review_results')
# Reuse this public example's existing annotation, not a newly guessed rectangle.
line=(base/'board_dataset/labels/test/37d8f468504ad26c.txt').read_text().splitlines()[0]
_,cx,cy,w,h=map(float,line.split())
box=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
config={'demo_not_site_policy':True,'food_model':'foodseg','boards':[{'id':'manual_example','xyxy_normalized':box,'allowed_foods':['meat']}]}
result=inspect_image(folder/'image.png',config,folder/'integration_check')
assert result['board_model']=='configured_region'
assert (folder/'integration_check/evidence.jpg').is_file()
record={'unit_tests': '11 passed (3 review tests + 8 rules)', 'hidden_Tk_initialization':'passed', 'visible_window_visual_review':'not performed', 'integration_output':str(folder/'integration_check'), 'status':result['overall_status'], 'source_annotation':'existing public test label; integration demonstration only; not new performance evidence'}
(base/'review_tool_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
