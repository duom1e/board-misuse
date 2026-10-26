> 最新训练已完成：experiments/redveg_contrast_frozen_backbone，512旧图+52原后厨+16对照图混训30轮freeze10。旧64图mAP50 .5777→.5666过回归门槛，但双机位操作门槛失败，B站18个菜GT仅TP1/FP18/FN17、16肉预测，红椒门槛失败。候选不采用，无训练后台；先读红椒混淆对照混训实验报告.md，勿重复此配置。

> 最新素材补充：新增下载BV1XUSbYiEUp红椒、BV14W41197Rd肉片肉丝，两视频24候选筛出16图、26食材粗框，已逐图叠框复核与SHA/YOLO范围审计。尚未新训练；每类一个新来源，类别/背景混淆需注意。先读红椒与肉类对照素材说明.md。

> 最新补充：B站BV14W4y1T7LC已完成25张ROI食材粗框复核，5张遮挡较重排除严格定位诊断。18个蔬菜GT上原foodseg TP0FP17FN18、第三十四轮候选TP2FP10FN16；多数预测把红色切碎辣椒报成肉。没有新训练。案板青框仍只是公共裁剪包络提案，不能当案板GT。先读B站备餐片段标注与识别核查.md。

> 2026-10-26接手核对更新：两段B站视频已实存D盘，别再要求用户下载抖音。3项自检已完成：规则8项OK、注册表6项、resume_selftest6帧读取/推理成功。原1000–2200探针混有非切配和装盒场景，不能整40帧自动打案板/食材真值；已逐张看4张对照表，初筛manifest在bilibili_prep_screening。另有完整历史第31–35轮食材多视角适配，已有肉/菜/空板素材，旧“只缺多种食材”“同机位就能用”概括过时。不要覆盖这些试验或误称已解决泛化。

# 交接提示词 · 给 Codex

项目：**后厨案板错用识别**（案板检测 + 板上食材判断 + 规则复核）。
工作目录：`D:\CodexProjects\board-misuse`。接手先跑完第九节的 3 条自检，再动模型。

---

## 一、30 秒看懂现状

| 能力 | 真实水平 |
| --- | --- |
| 菜品图分肉/菜（`foodseg`，2 类） | 测试 mAP50 0.591；conf 0.5 精确率 0.748 / 召回 0.424 |
| **标定过的固定机位**后厨找案板（`board_canteen`） | 同视频留出帧召回 0→**66.7%（0.5）/ 93.3%（0.25）**，旧能力保持 mAP50 0.926 |
| 默认案板模型（`board` = board_v2_real） | OI 案板测试 mAP50 0.912；家庭厨房精确率 0.44；**真实后厨 conf 0.5 召回 0** |
| 真实后厨判断食材 | conf 0.5 基本 0；conf 0.25 在第一视角备餐段有 19/40 帧能报出 |
| 生熟 / 视频事件 | 没有 |

**一句话**：案板侧已证明"给同机位真数据 + 混训就能用"；食材侧缺"多种食材 + 空板"的数据。

---

## 二、环境（踩过的坑都在这）

- Python：`D:\CodexProjects\board-misuse\work\board-env\Scripts\python.exe`（3.13.12 / torch 2.11.0+cu128 / ultralytics 8.4.188 / pyarrow 26 / yt-dlp 2026.10.21，GPU RTX 5070 Ti Laptop 12GB）
- 跑任何 ultralytics 脚本前设：`$env:YOLO_CONFIG_DIR="D:\CodexProjects\board-misuse\work\yolo-config"`（多数脚本已内置）
- **没有 ffmpeg**：所以 B 站下载要只取视频轨（`-f 30064/30032/30016`）；抽帧用 opencv，不依赖 ffmpeg
- **没有 selenium/playwright/pyppeteer**；本机浏览器 cookie 被 app-bound 加密，`yt-dlp --cookies-from-browser` 会失败
- PowerShell 控制台读 UTF-8 中文会乱码，**文件本身没坏**；打印加 `$env:PYTHONIOENCODING="utf-8"`，读文件用 read 工具
- 长训练用 `Start-Process` 起独立进程 + stdout 重定向（harness 的 background job 会随会话被杀）
- 网络：Open Images S3 ✅、hf-mirror ✅、B 站 API/CDN ✅、github 时通时断；huggingface.co ❌、Wikimedia/Openverse ❌、抖音 ❌（反爬）

---

## 三、模型注册表（`active_models.json` 是唯一权威）

| key | 权重 | 类别 | 备注 |
| --- | --- | --- | --- |
| `board`（默认） | `training_runs/board_v2_real/weights/best.pt` | cutting_board | 通用案板；真实后厨 conf 0.5 召回 0 |
| `foodseg`（默认食材） | `training_runs/foodseg_2cls/weights/best.pt` | meat, vegetable | 最好用的食材模型 |
| `vegetable` | `training_runs/vegetable_baseline/weights/best.pt` | vegetable | 老单类，兼容保留 |
| `food_type` | `training_runs/food_taxonomy_v1/weights/best.pt` | plant_food, animal_food | 对照用，已被 foodseg 超过 |
| `board_lvis`（可选） | `experiments/lvis_board_ccby_trial/training/weights/best.pt` | cutting_board | LVIS 来源候选，未达门槛 |
| `board_canteen`（可选） | `training_runs/board_v5_canteen_mix/weights/best.pt` | cutting_board | 某后厨机位适配，imgsz 960；同场景适配不是泛化 |

用法（单图/批量都支持 `--board-model`、`--food-model`）：

```powershell
Set-Location D:\CodexProjects\board-misuse\outputs
..\work\board-env\Scripts\python.exe .\check_board_misuse.py "D:\图.jpg" --allowed meat --food-model foodseg --board-model board_canteen
..\work\board-env\Scripts\python.exe .\batch_check_images.py "D:\某天图片" --allowed meat --food-model foodseg --output "D:\结果\20261026"
```

---

## 四、已经跑通的三条流水线（直接复用，别重写）

### 1) 拿到真实后厨视频 → 抽帧 → 跑模型

```powershell
$py="D:\CodexProjects\board-misuse\work\board-env\Scripts\python.exe"
Set-Location D:\CodexProjects\board-misuse\outputs
& $py .\search_bilibili.py                                  # 搜 37 条候选，写 bilibili_search.json
& $py -m yt_dlp -f 30064/30032/30016 -o "D:\CodexProjects\board-misuse\待导入视频\%(title).50s [%(id)s].%(ext)s" "https://www.bilibili.com/video/BVxxxx"
& $py .\import_video.py --frames 30                          # 整段：抽帧+检测+对照表
& $py .\probe_canteen2.py "<视频>" --start 1000 --end 2200 --frames 40   # 指定区间
```

产物：`outputs/imported/<视频名>/`（frames、detections.json、sheet_*.jpg）。对照表约定：**蓝=案板≥0.5，青=0.25–0.5，红=食材**。

### 2) 同机位域内适配（已证明有效，照抄）

第二十九轮的做法：
1. 用当前案板模型出提案（conf 0.1），取与"全帧中位框"IoU 最高者，离群回退中位框 → 得到每帧一个框；
2. 按时间切分：前 25 帧训练、后 15 帧评估；
3. **混训**：原训练集 + 这批帧（`run_canteen_mix.py`）；
4. 两个指标同时看：后厨留出帧召回 + Open Images 测试集是否退化。

**关键结论（务必遵守）**：
- 只用新数据微调 → 留出召回 100%，但旧测试 mAP50 从 0.912 崩到 **0.036**（灾难性遗忘）；
- **必须混训**：召回 66.7%/93.3%，旧测试 mAP50 0.926。

### 3) 先写判据再跑实验

所有实验都先写 `*_decision_rule.json`（阈值写死），再跑，再逐条判定。已两次拦住"看着 mAP 变好就宣布成功"。

---

## 五、数据资产（哪些已用尽）

| 数据 | 状态 |
| --- | --- |
| Open Images 案板图 196/10/24 | **已用尽**，不要再用它做"新"实验 |
| FoodSeg103 → `foodseg_two_class` 1708/254/248 | 已反复用于选型 |
| 商业厨房视频（HF `nova-dynamics/Chinese_Commercial_Kitchen_Manipulation_Dataset_Preview`，CC BY-NC 4.0） | 40 帧已抽，`real_canteen_probe/` |
| B 站两段后厨视频 | `待导入视频/`：炒锅线（竖屏）+ 第一视角备餐（横屏，含切菜段） |
| Pexels 481 张厨房/餐厅图 | 已用于域探测与负样本实验 |
| VISOR（EPIC-KITCHENS）第一视角 | 已用于适配实验，均失败（域太远） |

---

## 六、当前最值得做的三件事（按优先级）

1. **用 B 站第一视角备餐段做域内适配**（最直接）
   - 视频：`待导入视频\我在后厨的日常工作 [BV14W4y1T7LC].mp4`，备餐段约在 **帧 1000–2200**；
   - 案板小且被手遮，标"可见部分"；食材（西红柿、葱）可标；
   - 25 帧训练 / 15 帧评估 + 混训，指标看后厨召回 + 旧测试不退化。
2. **补食材数据缺口**：需要"多种食材 + 空砧板"。可从 B 站再收 3–5 段不同后厨/不同菜品，或按 `再要数据的说明.md` 里的文本找食堂授权拍摄。
3. **多场景验证**：现在所有"后厨可用"结论都来自**单场景**；换 2–3 个不同机位各采 20–30 帧混训，才能谈"可部署"。

---

## 七、不要做的事（红线）

- 不要伪造违规标签、准确率、"已上线"；不要把演示说成成果。
- 不要声称有大模型微调、生熟识别、视频事件处理——**都没有**。
- 所有"案板用途制度"都是演示假设，输出必须保留 `demo_not_site_policy=true`。
- 不要用已经用过的测试集反复调参；不要再用 Open Images 案板图做"新"实验。
- 单评审（助手目视）的结论要如实标注，不能写成专家标注。
- 不要为了"看起来更好"改判据；判据只能事前写。
- 不要再尝试抖音下载（反爬 + app-bound cookie 加密，已试尽）；B 站可用。

---

## 八、文件索引（关键）

| 文件/目录 | 作用 |
| --- | --- |
| `交付说明.md` | **给用户看的**能力边界与用法（先读这个） |
| `交接提示词.md` | 完整历史（30+ 轮，含每轮否决理由），本节是其 Codex 精简版 |
| `第一轮…第三十一轮实验报告.md` | 每轮数字、限制、结论 |
| `check_board_misuse.py` / `batch_check_images.py` / `misuse_rules.py` / `test_misuse_rules.py` | 推理入口、批量、规则引擎、8 项测试 |
| `import_video.py` / `导入视频.cmd` / `probe_canteen2.py` / `search_bilibili.py` | 视频抽帧、区间探针、B 站搜索 |
| `prepare_canteen_poscontrol.py` / `run_canteen_poscontrol.py` / `run_canteen_mix.py` | 同机位适配（纯微调 vs 混训）的完整实现 |
| `real_canteen_probe/` / `canteen_board_poscontrol/` / `board_dataset_v5/` | 商业厨房帧、标注、混训数据集 |
| `active_models.json` / `*_decision_rule.json` / `*_decision.json` | 注册表、判据、结论 |
| `再要数据的说明.md` | 给数据集作者/食堂的申请文本 |

---

## 九、接手自检（3 条，1 分钟）

```powershell
$py="D:\CodexProjects\board-misuse\work\board-env\Scripts\python.exe"
Set-Location D:\CodexProjects\board-misuse\outputs
$env:YOLO_CONFIG_DIR="D:\CodexProjects\board-misuse\work\yolo-config"
& $py .\test_misuse_rules.py            # 期望：8 项 OK
& $py -c "import json;d=json.load(open('active_models.json',encoding='utf-8'));print(list(d))"   # 期望 6 个 key
& $py .\probe_canteen2.py "D:\CodexProjects\board-misuse\待导入视频\我在后厨的日常工作 [BV14W4y1T7LC].mp4" --start 1000 --end 1100 --frames 6 --name selftest
```

第 3 条应输出 6 帧、并生成 `outputs/imported/selftest_prep/sheet_1.jpg`；看那张图应能看到第一视角备餐画面。自检通过就说明 GPU、模型、路径、数据都在位。
