"""Add verified local video and sample seeking to a completed evidence page."""
import argparse,hashlib,html,json,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('pipeline',type=Path);p.add_argument('--evidence-folder',default='evidence');a=p.parse_args();root=a.pipeline;folder=root/a.evidence_folder
 plan=json.loads((root/'pipeline_plan.json').read_text(encoding='utf-8'));events=json.loads((folder/'events.json').read_text(encoding='utf-8'));page=folder/'index.html';text=page.read_text(encoding='utf-8')
 if 'id="source-video"' in text:raise SystemExit('Timeline already installed; no overwrite')
 video=Path(plan['video']);digest=hashlib.sha256(video.read_bytes()).hexdigest()
 if digest!=plan['video_sha256']:raise ValueError('Video SHA mismatch; refuse to bind another source')
 if len(events['records'])!=text.count('<article '):raise ValueError('Card count mismatch')
 points=[r['time_seconds'] for r in events['records']]
 if any(not isinstance(t,(int,float)) or t<0 for t in points):raise ValueError('Valid sample timestamps required')
 target=folder/('source_video'+video.suffix.lower());shutil.copy2(video,target)
 if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Copied video mismatch')
 buttons=''.join(f'<button onclick="seekSample({t})">{t:g}秒</button>' for t in points)
 panel=f'''<section style="background:white;padding:20px;max-width:1000px;margin:auto;border-radius:12px"><h2>原视频与采样时间线</h2><p>点击时间查看前后动作。视频播放连续；识别仅发生在列出的采样点，不代表采样之间已被分析。</p><video id="source-video" controls preload="metadata" style="width:100%;max-height:520px" src="{html.escape(target.name)}"></video><p>请求范围 {plan['start']:g}—{plan['end']:g} 秒；采样间隔 {plan['interval']:g} 秒；实际末点 {points[-1]:g} 秒。</p>{buttons}<p>播放视频只是供人工复核，不会触发新的模型识别，也不推断接触、清洗或生熟。</p></section><script>function seekSample(t){{const v=document.getElementById('source-video');const seek=()=>{{v.currentTime=t;v.scrollIntoView({{behavior:'smooth',block:'center'}})}};if(v.readyState>=1)seek();else v.addEventListener('loadedmetadata',seek,{{once:true}})}}</script>'''
 # Existing generated pages omit a body tag; insert after the closing header.
 assert '</header>' in text;text=text.replace('</header>','</header>'+panel,1)
 offset=0
 for t in points:
  pos=text.index('<article ',offset);end=text.index('</h2>',pos)+5;text=text[:end]+f'<button onclick="seekSample({t})">跳到原视频 {t:g} 秒</button>'+text[end:];offset=end+70
 page.write_text(text,encoding='utf-8')
 (folder/'timeline_binding.json').write_text(json.dumps(dict(source_video=str(video),sha256=digest,local_copy=target.name,sample_seconds=points,interval=plan['interval'],continuous_analysis=False,training_performed=False),indent=2),encoding='utf-8');print('Bound verified video to',len(points),'sample cards')
if __name__=='__main__':main()
