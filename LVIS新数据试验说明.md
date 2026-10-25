# LVIS新数据试验说明

本轮继续处理案板定位瓶颈，尝试新的、有实例标注的公开来源，不重复从已用过的Open Images案板子集选型。

## 数据来源与筛选

[LVIS官方代码与数据说明](https://github.com/lvis-dataset/lvis-api)提供大词汇实例分割标注。本次从官方v1标注中提取 `chopping_board`（ID253，包含cutting_board同义词）的边界框；排除该类别被标为not-exhaustive的图片。

图像许可逐图依据LVIS元数据，只保留license ID4，即CC BY 2.0，不将其他许可图像混入本子集。LVIS标注许可与图像许可分开处理；[作者关于许可的说明](https://groups.google.com/g/lvis-dataset/c/JeWfhD3vioM)指出标注为CC BY 4.0，而图像来自COCO并有单独条款。

原标注中，训练部分有113张符合该图像许可的案板正样本，验证部分有16张。结合来源明确标为该类不存在的负样本，共320张候选。根据之前实际使用图片的Flickr来源ID排除1张，下载后又与5631张既有独特图片做文件哈希和dHash筛查，最终保留319张新来源图片。

图片的原COCO URL、实际S3下载URL、Flickr原图链接、许可ID、来源框和显式negative-category记录保存在 `board_lvis_ccby/manifest.json`。统一来源表为 `attribution.csv`。LVIS未提供逐图作者名字，表中如实标为未提供，不冒充已完成独立版权核验。图片只存于本地项目，未对外发布。

## 集合与数量

| 集合 | 图片 | 案板正样本图片 | 明确负样本图片 | 案板框 |
|---|---:|---:|---:|---:|
| train | 466 | 288 | 178 | 323 |
| val | 57 | 23 | 34 | 28 |
| test | 38 | 15 | 23 | 19 |

训练部分包含历史Open Images训练集242张及新LVIS训练224张。验证来自LVIS官方train中按固定seed分出的20%；测试来自LVIS官方val，不将原Open Images验证或测试图混入新来源评估。

预检查没有发现跨集合字节重复、解码RGB重复或dHash<=4候选。缺少完整的拍摄者/场景分组，照片ID和视觉筛查不能证明不同厨房或不同家庭独立。这仍是公开家庭厨房、食材/菜品照片，不是食堂监控数据。

只查看了训练标注预览，未在选择训练参数前查看测试预测或按测试结果调参。旧公开测试仅用作预先定义的回归检查，不作为新泛化成绩。

另有预训练来源限制：本轮38张测试照片中，23张底图来自COCO train2017（8张案板正图、15张负图），15张来自COCO val2017（7正、8负）。初始通用YOLO权重的训练配置是coco.yaml。因此“新来源测试”指本项目此前未用过这些案板标注，不代表底层预训练完全没见过图像。比较报告会另列两个来源组的固定阈值结果，这只是来源诊断，不新增验收条件。记录见lvis_pretraining_overlap_note.json。

## 训练与判据

- 初始权重：当前注册的board_v2_real。
- 50轮，640输入，batch8，AdamW lr0=0.0005，lrf0.1，mosaic0.3，scale0.2，不冻结骨干，seed42。
- 固定confidence0.5、匹配IoU0.5；不在本轮测试集上扫描阈值。
- 绝对与相对条件在训练前写入 `lvis_board_decision_rule.json`，其哈希已放入计划并随实验锁定。
- 新来源P至少0.70、R至少0.50、mAP50至少0.60；F1比参考提高至少0.05，同时P/R退步和负图误报增量有上限。
- 旧样本回归检查限制mAP50及固定阈值召回下降。不满足全部条件时保留当前默认模型。

## 文件与恢复入口

- 数据：`board_lvis_ccby/`。
- 准备：`prepare_lvis_boards.py`，筛选元数据在 `work/datasets/lvis/lvis_board_candidates.json`。
- 训练计划：`lvis_board_plan.json`；预写判据：`lvis_board_decision_rule.json`。
- 实验目录：`experiments/lvis_board_ccby_trial/`；预检查：`experiments/lvis_board_preflight/`。
- 训练PID与日志：`lvis_board_process.json`、`lvis_trial_training_stdout.txt`、`lvis_trial_training_stderr.txt`。
- 训练采用隐藏独立进程，不能因为聊天中断就假设已经停止；续接前先查PID、report.json和results.csv，禁止重复启动。
- 完成后运行一次 `compare_lvis_board_trial.py`，对同一新来源测试集比较参考与候选，并生成完整对照图及逐条判据结果；该脚本拒绝覆盖已完成比较。

训练和对照已完成，完整结果见 `第十五轮实验报告.md`、`lvis_board_comparison.json`。候选P0.900、R0.474、mAP50为0.405；未满足全部预写条件，默认权重未替换，候选仅登记为board_lvis显式试用。进程记录已标为completed。38张负图中包含与厨房无关的生活/户外场景，不能据此宣称后厨精确率90%。
