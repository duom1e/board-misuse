"""Export only project text/code; never traverse local data/model directories."""
import json,re,hashlib,zipfile,importlib.metadata
from pathlib import Path
B=Path(__file__).resolve().parent;O=B/'github_export_20261029'
def scrub(s):
 s=re.sub(r'C:[\\/]Users[\\/]13442', '<USER_HOME>',s,flags=re.I)
 return s
if __name__=='__main__':
 O.mkdir(exist_ok=False)
 files=[]
 excluded={'环境检查结果.md','D盘存储位置.md','_check_chrome_cookies.py','_douyin_cookie_attempt.py','_douyin_probe.py','copy_browser_cookies.py','grab_douyin.py'}
 for p in sorted(B.iterdir()):
  if p.is_file() and p.suffix.lower() in {'.py','.md'} and p.name not in excluded:
   (O/p.name).write_text(scrub(p.read_text(encoding='utf-8-sig')),encoding='utf-8')
 for name in ['requirements-lock.txt','规则配置示例.json','experiment_new_photos.example.json','lvis_board_decision_rule.json','lvis_pretraining_overlap_note.json']:
  (O/name).write_text(scrub((B/name).read_text(encoding='utf-8-sig')),encoding='utf-8')
 registry=json.loads((B/'active_models.json').read_text(encoding='utf-8-sig'));manifest={}
 for k,v in registry.items():
  p=Path(v['weights']);rel=p.relative_to(B).as_posix()
  manifest[k]={'weights':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'included_in_git':False,'classes':v.get('classes'),'imgsz':v.get('imgsz'),'status':v.get('status')};v['weights']=rel
 (O/'active_models.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
 (O/'model_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 for name in ['configure_local_models.py','打开图片复核工具.cmd']:
  (O/name).write_bytes((B/'github_export_20261025'/name).read_bytes())
 (O/'requirements-video.txt').write_text('yt-dlp=='+importlib.metadata.version('yt-dlp')+'\n',encoding='utf-8')
 (O/'.gitignore').write_text('''__pycache__/
*.pyc
.venv/
.env
*.pt
*.onnx
*.mp4
*.mov
*.avi
*.parquet
*.zip
*.log
*.cache
work/
*_dataset*/
training_runs/
experiments/
imported/
待导入视频/
review_results/
annotation_pool/
annotation_audits/
batch_results/
misuse_results/
github_export_*/
''',encoding='utf-8')
 (O/'README.md').write_text('''# 案板错用识别实验项目

案板检测、肉/蔬菜检测和用途规则复核的本地研究原型。实现为YOLO检测加规则，**尚未做视觉语言大模型微调，尚不能可靠自动识别真实后厨案板错用**。不能判断生熟、清洁历史或实际接触；未检出不代表空板或合规。

## 最新状态（2026-10-26）

- 默认食材模型仍为foodseg_2cls；历史菜品测试mAP50约0.591，不能视为真实后厨准确率。
- 新增商业厨房多视角和B站切配片段研究，已完成局部食材标注、固定阈值选权重及混训实验。公开视频的可访问性不等于创作者授予训练或发布许可。
- 最近发现切碎红椒常被误认成肉。新增两段对照视频，人工复核16张图、26个食材粗区域框，均由助手单评审，非专家标注。
- 最新30轮冻结前段骨干混训：580张独特训练图、612曝光条目。旧64张验证mAP50由0.5777降至0.5666，通过预设回归门槛；双机位操作门槛及红椒诊断门槛失败。B站18个蔬菜区域仅匹配1处，16个肉类预测。**候选未采用，默认模型未替换**。
- 多数片段已经用于开发，近邻帧强相关。上述数字均不是独立跨厨房或最终测试成绩，也不证明部署可用。

最新详述：[红椒混淆对照混训实验报告](红椒混淆对照混训实验报告.md)、[红椒与肉类对照素材说明](红椒与肉类对照素材说明.md)、[B站备餐片段标注与识别核查](B站备餐片段标注与识别核查.md)。完整历史在[交接提示词](交接提示词.md)，另有[精简交接](交接提示词-Codex.md)。轮次编号存在不同历史分支重复，以文件内容和最新交接更新为准。

## 运行入口

- `review_app.py`：本地图片与逐板用途人工复核。
- `check_board_misuse.py`、`batch_check_images.py`：单图/批量疑似冲突提示。
- `import_video.py`、`probe_canteen2.py`：本地视频抽帧和检测；检测框不是训练真值。
- `prepare_manual_dataset.py`、`run_experiment.py`：数据审计、来源组划分和实验管线。
- `run_redveg_contrast_trial.py`：最新失败实验的预锁训练方案；依赖本机数据和已有中间文件，不能只克隆代码就复现数据部分。

模型配置和SHA见`model_manifest.json`。仓库不含权重，需另行取得有权使用的权重，按对应路径放置后运行`python configure_local_models.py`。部分研究脚本仍有D盘路径，换机器需要调整；本次发布没有改动原电脑的数据或模型。

Python 3.13及原Windows/CUDA依赖见`requirements-lock.txt`；视频下载依赖另见`requirements-video.txt`。没有跨平台一键安装保证。下载视频只取视频轨，抽帧使用OpenCV，不要求本地ffmpeg。

本次同步核对的测试：

```console
python -m unittest test_misuse_rules test_operational_selection test_manual_dataset test_annotation_review test_experiment_pipeline test_review_sessions
```

软件测试与模型准确率分开记录。用途规则必须来自现场规定；演示规则保留`demo_not_site_policy=true`，不能据此给实际厨房判罚。

## 数据与发布范围

本仓库同步代码、文字报告、交接和模型路径/SHA清单；**不含原图、视频、数据集、逐图私人复核数据、训练缓存、原始日志或权重**。新B站素材及商业厨房数据的原始文件、模型均留本机。第三方数据、基础模型和依赖各有独立条款，本仓库不替它们授予许可。

报告中的本机路径、配套图片和实验JSON可能不在仓库；不要把缺少这些数据的源码快照当作完整可运行模型包。后续重点为细碎红椒/肉末对照、标注一致性、保留旧能力的受控实验，以及真正独立操作/厨房的评估。
''',encoding='utf-8')
 for p in sorted(O.iterdir()):
  if p.is_file():
   text=p.read_text(encoding='utf-8-sig').replace('\r\n','\n');p.write_text(text,encoding='utf-8',newline='\n');data=p.read_bytes();files.append({'path':p.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha':hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()})
 # No model/data binaries or nested data directories can enter this export.
 assert all(Path(f['path']).suffix.lower() in {'.py','.md','.json','.txt','.cmd',''} for f in files)
 assert not any(p.is_dir() for p in O.iterdir())
 meta={'folder':str(O),'files':files,'excluded':'images/videos/datasets/weights/logs/private review data','weights_uploaded':False}
 (B/'github_latest_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'files':len(files),'bytes':sum(f['bytes'] for f in files),'folder':str(O)}))
