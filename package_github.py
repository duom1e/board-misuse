import hashlib,json,re,shutil,zipfile
from pathlib import Path
src=Path(r'D:\CodexProjects\board-misuse\outputs')
root=src/'github_export_20261025'
root.mkdir(exist_ok=False)
# Preserve source layout: existing scripts import their sibling modules.
for p in src.iterdir():
    if p.is_file() and (p.suffix=='.py' or (p.suffix=='.md' and p.name not in ['交接提示词.md','D盘存储位置.md','环境检查结果.md'])):
        text=p.read_text(encoding='utf-8-sig')
        text=text.replace(r'<LOCAL_WORKSPACE>','<LOCAL_WORKSPACE>')
        text=text.replace('<LOCAL_WORKSPACE>','<LOCAL_WORKSPACE>')
        (root/p.name).write_text(text,encoding='utf-8')
for name in ['requirements-lock.txt','规则配置示例.json','experiment_new_photos.example.json','lvis_board_decision_rule.json','lvis_pretraining_overlap_note.json']:
    shutil.copyfile(src/name,root/name)
registry=json.loads((src/'active_models.json').read_text(encoding='utf-8-sig'))
manifest={}
for key,value in registry.items():
    p=Path(value['weights'])
    manifest[key]={'weights':str(p.relative_to(src)).replace('\\','/'),'status':value.get('status'),'imgsz':value.get('imgsz'),'classes':value.get('classes'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,'included_in_git':False}
    value['weights']=manifest[key]['weights']
(root/'active_models.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'model_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'.gitignore').write_text('__pycache__/\n*.pyc\n.venv/\n.env\n*.pt\n*.onnx\n*.zip\nwork/\n*_dataset*/\ntraining_runs/\nexperiments/\nreview_results/\nannotation_pool/\nannotation_audits/\nbatch_results/\nmisuse_results/\n',encoding='utf-8')
(root/'configure_local_models.py').write_text('''"""Resolve model paths after placing local weights at the paths in model_manifest.json."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
path=root/'active_models.json'
registry=json.loads(path.read_text(encoding='utf-8-sig'))
for entry in registry.values():
    weight=Path(entry['weights'])
    if not weight.is_absolute(): entry['weights']=str(root/weight)
path.write_text(json.dumps(registry,ensure_ascii=False,indent=2),encoding='utf-8')
print('Registry configured. Missing weights:')
for key,entry in registry.items():
    if not Path(entry['weights']).is_file(): print(key,entry['weights'])
''',encoding='utf-8')
(root/'打开图片复核工具.cmd').write_text('@echo off\ncd /d "%~dp0"\npython review_app.py\nif errorlevel 1 pause\n',encoding='ascii')
readme='''# 案板错用识别实验项目

本地图片识别与人工复核原型：案板检测、肉类/蔬菜检测、明确用途规则、证据图输出、人工案板标注及按来源组整理、训练与数据审计。

这是 YOLO 检测加规则的实验项目，尚未训练视觉语言大模型；不能判断食物生熟、清洁历史或实际接触，也不能输出“确认合规”。

## 当前结果

新增LVIS案板候选在38张新来源图片（19个案板框）上，固定阈值精确率0.900、召回率0.474、mAP50 0.405；未达到预先规定的全部条件，原默认模型保留。该集合不是食堂监控，部分底图可能已在COCO预训练中出现。不要把它当现场准确率。

640/960/1280及局部放大诊断未达到预定改善条件。人工样本池目前为空。软件共25项检查通过，与模型准确率分别记录。第十五至十七轮报告记录最新结论；早期报告保留历史状态。

## 内容与启动

- `review_app.py`：打开本地图、手工多框、明确统一用途后识别、导出人工标注。
- `batch_check_images.py`：文件夹批量识别，隔离坏图，保存证据和模型快照。
- `prepare_manual_dataset.py`：样本完整性、去重候选、按来源组显式划分，默认不启动训练。
- `run_experiment.py`：计划锁定、数据审计、训练、验证校准和固定规则评估。
- `图片复核工具使用说明.md`、`人工样本整理说明.md`、`训练管线使用说明.md`：操作及限制。

建议使用Python 3.13。`requirements-lock.txt`是原Windows/CUDA环境记录，torch的CUDA版本需使用适合本机的安装来源；并非跨平台一键安装保证。

仓库不含数据集和模型二进制。模型路径与SHA256见`model_manifest.json`；将本地权重放到对应路径，运行`python configure_local_models.py`后再启动`python review_app.py`。部分历史脚本保留D盘工作路径，换电脑需调整。原电脑仍可使用原D盘项目环境。

纯软件检查：

```console
python -m unittest test_manual_dataset test_annotation_review test_misuse_rules test_experiment_pipeline
```

Tk隐藏窗口初始化已检查；可见窗口完整视觉/鼠标验收尚未完成。规则只用于疑似冲突提示，必须由人工复核。

## 数据和许可

未上传原图、数据集、个人复核图片、原始日志或训练缓存。Open Images、FoodSeg103、LVIS及基础模型各有独立条款，使用或再分发前需核对各自许可。未为第三方素材或基础模型授予新许可。

此仓库是截至2026-10-25的成果快照；报告中的本机路径和历史产物不一定包含在仓库中。权重保存在原电脑本地完整备份中。
'''
(root/'README.md').write_text(readme,encoding='utf-8')
# Keep a locally complete backup of the selected runnable models, without publishing image data.
archive=src/'board-misuse-20261025-backup.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in root.rglob('*'):
        if p.is_file(): z.write(p,'board-misuse/'+p.relative_to(root).as_posix())
    for key in ['board','board_lvis','foodseg','vegetable']:
        p=Path(json.loads((src/'active_models.json').read_text(encoding='utf-8-sig'))[key]['weights'])
        z.write(p,'board-misuse/'+p.relative_to(src).as_posix())
files=[{'path':p.relative_to(root).as_posix(),'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file()]
(src/'github_upload_manifest.json').write_text(json.dumps({'folder':str(root),'archive':str(archive),'files':files},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'files':len(files),'bytes':sum(p['size'] for p in files),'archive_bytes':archive.stat().st_size,'folder':str(root)},ensure_ascii=False))
