"""One-shot CLI integration checks using saved responses; never invokes VLM."""
import json,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent;E=B/'experiments/video_entrypoint_v1/integration_checks'
def main():
    E.mkdir(exist_ok=True);assert not (E/'checks.json').exists()
    original=json.loads((E.parent/'replay.json').read_text(encoding='utf-8'));video=B.parent/'待导入视频/paired_BV1qT4y127Go.mp4';checks=[]
    cases=[('wrong_video',original|{'video_sha256':'0'*64},'Replay does not match'),('missing_frame',original|{'responses':{}},'Replay is missing')]
    for name,data,error in cases:
        replay=E/(name+'.json');replay.write_text(json.dumps(data),encoding='utf-8');out=E/name
        r=subprocess.run([sys.executable,str(B/'review_board_video.py'),str(video),'--end','0','--replay',str(replay),'--output-dir',str(out)],capture_output=True,text=True,encoding='utf-8',errors='replace')
        assert r.returncode!=0 and error in r.stderr and not out.exists();checks.append({'case':name,'passed':True})
    data=json.loads(json.dumps(original));data['responses']['0']['image_sha256']='0'*64;replay=E/'wrong_image.json';replay.write_text(json.dumps(data),encoding='utf-8');out=E/'wrong_image'
    r=subprocess.run([sys.executable,str(B/'review_board_video.py'),str(video),'--end','0','--replay',str(replay),'--output-dir',str(out)],capture_output=True,text=True,encoding='utf-8',errors='replace')
    assert r.returncode==0,r.stderr
    row=json.loads((out/'results.json').read_text(encoding='utf-8'))['rows'][0]
    assert row['board_state']=='unknown' and not row['inference_performed'] and 'raw_response' not in row and 'pixel_boxes' not in row
    assert row['frame_error']['code']=='replay_error';checks.append({'case':'wrong_image_hash_kept_unknown','passed':True})
    (E/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print('3 CLI integration checks passed, no VLM calls.')
if __name__=='__main__':main()
