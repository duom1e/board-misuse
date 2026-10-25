"""Summarize the completed, pre-registered LVIS data trial without relabelling failure as success."""
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent

def main():
    d=json.loads((BASE/'lvis_board_comparison.json').read_text(encoding='utf-8'))
    rows=[]
    for title,key in [('参考模型','fresh_reference'),('LVIS候选','fresh_candidate')]:
        r=d[key];m=r['fixed']['micro'];rows.append(f"| {title} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.1%} | {m['recall']:.1%} | {m['f1']:.3f} | {r['framework']['map50']:.3f} | {r['negative_probe']['false_boxes']} |")
    checks='\n'.join(f"| {c['criterion']} | {c['observed']:.4f} | {c.get('minimum',c.get('maximum')):.4f} | {'通过' if c['passed'] else '未通过'} |" for c in d['checks'])
    strata=[]
    for name,r in d['pretraining_source_strata'].items():
        a=r['reference']['micro'];b=r['candidate']['micro']
        strata.append(f"| {name} | {r['images']} | {r['positive_images']} | {a['precision']:.1%} / {a['recall']:.1%} | {b['precision']:.1%} / {b['recall']:.1%} |")
    text=f'''# 第十五轮实验报告：补充LVIS来源，误报减少，召回仍不足

## 结果与决定

新增独立来源的案板标注后，在本项目未用于案板训练/调参的38张LVIS图片上，固定confidence0.5、匹配IoU0.5的精确率从46.7%提高到90.0%，召回从36.8%提高到47.4%，负图误报框从5个降至0个。

**没有通过全部预先写定的验收条件。** mAP50为0.405，低于预定0.60；固定阈值召回0.474，低于0.50；管线额外规定的mAP50–95为0.242，低于0.30。因此没有改低门槛宣布达标，默认board仍为board_v2_real。

候选保留为 `active_models.json` 的 `board_lvis`，仅通过 `--board-model board_lvis` 显式试用。foodseg与原有默认项保持不变。没有生熟、清洗或视频事件能力，也没有大模型微调。

## 新数据与划分

来源为[LVIS官方资料](https://github.com/lvis-dataset/lvis-api)。从官方v1标注提取chopping_board ID253，只保留图像许可元数据ID4（CC BY2.0），排除该类not-exhaustive图片；负样本必须有明确的neg_category_ids记录。

320张候选按既有Flickr来源排除1张，319张新图再与5631张既有独特图片做SHA/dHash检查。新LVIS train按seed分成224训练、57验证，LVIS官方val的38张作为新来源测试。另保留242张历史训练图，旧验证/测试未混入新来源训练。

| 集合 | 图片 | 正图 | 负图 | 框 |
|---|---:|---:|---:|---:|
| train | 466 | 288 | 178 | 323 |
| val | 57 | 23 | 34 | 28 |
| test | 38 | 15 | 23 | 19 |

预检查未发现跨集合字节、解码RGB重复或dHash<=4候选。没有完整拍摄者/场景分组，不能证明场景完全独立。署名、原图链接、许可元数据和来源标注留在board_lvis_ccby目录。LVIS元数据没有作者姓名，表中如实记录“未提供”，未声称独立核验所有图片权利。

## 新来源固定阈值对照

| 模型 | TP | FP | FN | 精确率 | 召回率 | F1 | mAP50 | 23张负图误报框 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

TP/FP/FN按类别、置信度排序、一对一IoU匹配统计。P/R不是框架曲线选点口径，mAP采用框架标准口径。

这批负图包括人物、运动、户外、食物与家庭场景，**不是23张真实后厨负图**。90.0%是这10个预测框中9个匹配标注的结果，不能称为后厨识别准确率。

## 旧样本回归检查

在反复用于开发的24张历史案板图上，mAP50从0.912变为0.941；固定阈值P/R从90.5%/76.0%变为91.7%/88.0%。它只用于预写的回归检查，不能作为新的独立提升证据。

## 来源与预训练限制

本轮38张底图中23张来自COCO train2017，通用YOLO初始权重用coco.yaml预训练，可能已经见过这些图用于其他类别。因而“新来源测试”只指本项目此前没有使用这些案板标注，不是全球未见图像。

以下是来源诊断，不是新增的验收标准：

| COCO底图来源 | 图片 | 正图 | 参考P/R | 候选P/R |
|---|---:|---:|---:|---:|
{chr(10).join(strata)}

val2017来源子组仅15张、7正图、11个案板框；候选仍漏掉7个框，小样本不支持强泛化结论。

## 预先判据逐条记录

全部门槛在训练前写入lvis_board_decision_rule.json，并将哈希放入冻结计划。没有按测试结果修改门槛。

| 条件 | 实际值 | 最低/最高门槛 | 结果 |
|---|---:|---:|---|
{checks}

此外，管线绝对条件中mAP50–95和召回也未达标。详见实验decision.json。

## 目视检查与软件接入

查看了第一张完整12图对照表，以及候选唯一的未匹配预测：lvis_000000287667中把木桌面框成案板，而真实板位于画面上方，IoU约0.204。源图中的远景小板也仍有漏检。这些现象说明“物体出现位置”和“木材纹理”仍会干扰模型。

对照表按测试文件顺序完整输出，未挑选有利案例；其余对照表可在comparison_sheets中查看。人工目视仅为助手单评审，定量成绩以来源标注和固定匹配为准。

新候选已接入单图、批量及演示脚本。用既有葱花案板图做软件演示时，候选可以自动定位，肉类专用假设触发疑似冲突，蔬菜专用假设保留整图复核。它是既有公开图、假设规则，不是违规性能测试。坏图隔离、重名路径、实际权重记录的集成检查已再次通过，规则8项检查通过。

## 运行方式

```powershell
Set-Location D:\\CodexProjects\\board-misuse\\outputs
..\\work\\board-env\\Scripts\\python.exe .\\batch_check_images.py "D:\\待检图片" --allowed meat --food-model foodseg --board-model board_lvis
```

仅在调用者明确这组案板允许肉类时使用该规则。模型候选只输出疑似结果，不能自动确认污染或违规。

## 文件

- 数据来源与构建：LVIS新数据试验说明.md、prepare_lvis_boards.py、board_lvis_ccby/。
- 预写规则与训练计划：lvis_board_decision_rule.json、lvis_board_plan.json。
- 训练：experiments/lvis_board_ccby_trial/training/weights/best.pt，50轮约6分钟；完整日志在lvis_trial_training_stdout.txt。
- 定量对照：lvis_board_comparison.json；可复核图：experiments/lvis_board_ccby_trial/comparison_sheets/。
- 木桌误检图：lvis_unmatched_prediction.jpg。
- 候选接入：lvis_board_model_record.json；注册表原备份active_models_before_lvis_candidate.json。
- 批量软件检查：batch_results/lvis_candidate_smoke/summary.md。
- 演示：misuse_results/lvis_candidate_summary.json。

## 下一步

可以继续解决远景小板和木桌面混淆，优先补充这两类的清晰标注与不同采集场景。新公开来源已补齐一部分素材与训练，但真实目标后厨的可靠性仍无法从生活/菜品照片推断。后续改进以验证集为开发依据；本轮38张图已经用于判断，不能再冒充全新测试集。
'''
    (BASE/'第十五轮实验报告.md').write_text(text,encoding='utf-8')
    print(BASE/'第十五轮实验报告.md')

if __name__=='__main__':main()
