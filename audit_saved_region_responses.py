"""Offline parser audit of stored outputs; no inference or metric rewriting."""
import json,hashlib
from pathlib import Path
from region_response_validation import validate_region_response
B=Path(__file__).resolve().parent;E=B/'experiments/response_validation_v1'
def main():
    sources=['qwen4b_new_sources_v1','qwen4b_paired_video_v1','qwen4b_same_kitchen_v1','qwen4b_temporal_presence_v1'];rows=[];hashes={}
    for folder in sources:
        path=B/'experiments'/folder/'results.json';digest=hashlib.sha256(path.read_bytes()).hexdigest();hashes[str(path)]=digest
        for r in json.loads(path.read_text(encoding='utf-8'))['rows']:
            validation=validate_region_response(r['response'])
            rows.append({'experiment':folder,'case':r['case'],'response_sha256':hashlib.sha256(r['response'].encode()).hexdigest(),'old_valid':'pixel_boxes' in r,'new_valid':validation['valid'],'errors':validation['errors'],'board_state':validation['board_state']})
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    result={'scope':'Offline audit of stored, correlated development responses. No new recognition test, inference or historical metric changes.','total_responses':len(rows),'valid':sum(r['new_valid'] for r in rows),'validation_disagreements':sum(r['old_valid']!=r['new_valid'] for r in rows),'source_file_sha256':hashes,'rows':rows}
    (E/'saved_response_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print({k:result[k] for k in ['total_responses','valid','validation_disagreements']})
if __name__=='__main__':main()
