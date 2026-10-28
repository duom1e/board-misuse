"""Add explicit human review notes; never change model observations."""
import argparse,hashlib,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();page=a.evidence/'index.html';events=a.evidence/'events.json';text=page.read_text(encoding='utf-8')
 if 'id="review-export"' in text:raise SystemExit('Review controls already installed')
 data=json.loads(events.read_text(encoding='utf-8'));digest=hashlib.sha256(events.read_bytes()).hexdigest();count=len(data['records'])
 if text.count('<article ')!=count:raise ValueError('Evidence card mismatch')
 controls='''<section style="padding:20px;border:1px solid #bbc"><h3>人工复核（独立记录）</h3><p>仅评价画面与识别结果，不代表确认违规。</p><label>判断 <select class="review-decision"><option value="pending">尚未复核</option><option value="observation_supported">识别基本正确</option><option value="false_region">存在误框或错误类别</option><option value="uncertain">无法判断</option></select></label><label> 备注 <textarea class="review-note" rows="3" placeholder="例如：候选框包含手部；食材被举起，接触无法确认" style="display:block;width:95%"></textarea></label></section>'''
 text=text.replace('</article>',controls+'</article>')
 panel='''<section style="background:white;padding:20px;max-width:1000px;margin:auto"><h2>人工复核记录</h2><p>先填写各证据卡片的复核栏，再导出JSON保存。记录仅在当前页面内存中；刷新前请导出。不会修改模型输出，也不会自动回灌训练。</p><button id="review-export" onclick="exportReviews()">导出人工复核JSON</button><p id="review-message" role="status"></p></section>'''
 script='''<script>function exportReviews(){const records=Array.from(document.querySelectorAll('article')).map((a,i)=>({index:i,decision:a.querySelector('.review-decision').value,note:a.querySelector('.review-note').value}));const output={schema_version:1,source_events_sha256:SOURCE_HASH,created_at:new Date().toISOString(),scope:'human observation review only; not confirmed misuse, training labels or site policy',records};const blob=new Blob([JSON.stringify(output,null,2)],{type:'application/json;charset=utf-8'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='human_review.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('review-message').textContent='已请求下载，请确认文件已保存。';}</script>'''.replace('SOURCE_HASH',json.dumps(digest))
 text=text.replace('</header>','</header>'+panel,1)+script;page.write_text(text,encoding='utf-8')
 (a.evidence/'review_binding.json').write_text(json.dumps(dict(source_events_sha256=digest,cards=count,model_results_changed=False,review_storage='memory_until_export',browser_interaction_tested=False),indent=2),encoding='utf-8');print('Added review controls:',count)
if __name__=='__main__':main()
