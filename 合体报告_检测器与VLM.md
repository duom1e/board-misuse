# 合体：检测器线 + VLM 线合成一个入口（2026-11-01）

## 一、为什么要合

两条线各自被证明"只擅长一半"：

| 线 | 擅长的 | 实测证据 | 不擅长的 |
| --- | --- | --- | --- |
| 检测器（YOLO11，第 28/30 次那批） | 食物是什么（meat / vegetable）、框相对紧 | 第 28 次：旧 mAP50 0.5915、食材 16/19 | 真实后厨判"有没有板"：conf 0.5 召回 0/40 |
| VLM（Qwen3-VL-4B 单问） | 有没有案板 | 49 帧：可解析 **49/49**、有板 **2/2**、误报 1/39 | 框常常覆盖整幅（2/6 候选），不能直接算位置 |

所以合体的目标不是"更准"，而是：**存在性用 VLM、食物用检测器、位置用能用的那个框，并且把不可用的情况明确说出来。**

## 二、融合规则（写死在文件头，未事后调整）

1. **板存在性以 VLM 为准**；若 VLM 说"没有"但检测器案板模型 conf≥0.5 检出 → 标 `uncertain_vlm_no_yolo_yes`，不采信任一方；
2. **板位置**：优先 VLM 框，且必须过 `box_quality`；VLM 框过粗 → **退到检测器框**；两者都过粗 → `position_usable=false`，提示人工框定；
3. **食物**：只用检测器框（conf≥0.25），位置归属走 `food_region_gate`，并受 `position_usable` 护栏（粗框时不会给出"食物在板上"的结论）；
4. **永远是 `needs_review`**，不判定合规；结果里逐条标注证据来自哪条线。

## 三、两个实跑（同一入口，两种画面）

**A. 有板帧**（白色案板近景，E66 010s）

| 项 | 结果 |
| --- | --- |
| VLM 板判断 | `candidate_present`，但框 `[[0,0,998,998]]`（面积 0.996）→ **过粗** |
| 检测器板框 | 1 个，`[[1.5,61.4,850.5,717.8]]`（面积约 0.60）→ 也到过粗边界 |
| 融合结果 | `present_vlm_box_coarse_using_yolo`，`position_usable=false` |
| 食物归属 | 5 个蔬菜框，全部 `uncertain / board_box_too_coarse` |

→ 这正是想要的：**承认"看到板了，但位置不可用"**，而不是拿整幅框算出一个假的"食物在板上"。

**B. 无板帧**（洗菜机，Fj 052s）

| 项 | 结果 |
| --- | --- |
| VLM | `no_board_detected` |
| 检测器 | 案板 0 框、食物 0 框 |
| 融合结果 | `absent`，`position_usable=false (no_boxes)` |
| 食物归属 | 无 |

→ 两条线一致沉默，结论是"没有报出案板"，但报告里仍写明**这不等于画面无板**。

## 四、用法

```powershell
..\work\vlm-env\Scripts\python.exe .\review_combined.py "D:\图.jpg" --output-dir "D:\结果\case01"
# 没有 GPU 时：加 --no-vlm，只跑检测器线并交叉验证
```

输出：`result.json`（含 `lines.detector` 与 `lines.vlm` 两条原始证据 + `fusion` 决策记录）、`review.jpg`（黄=检测器板、青=VLM 板、红=食物）、`review_report.md`（人读版）。

## 五、限制（必须一起读）

- 有板样本合计仍只有 2 帧，**这不是准确率**；
- 融合不会提升任何单条线的能力，只是把"用谁的判断、用谁的框"规则化；
- 检测器线在真实后厨判"有没有板"依旧不可靠，只做交叉验证；
- VLM 的框过粗问题没有解决，只是被拦住了；
- 结果永远 `needs_review`、`compliance_determined=false`。

## 六、文件

| 文件 | 说明 |
| --- | --- |
| `review_combined.py` | 合体入口（VLM + 检测器 + 规则层） |
| `experiments/combined_v1/with_board/`、`.../no_board/` | 两次实跑的 result.json / review.jpg / review_report.md |

## 七、批量验证（49 帧）与一条被数据否掉的规则

**跑法**：`run_combined_batch.py`（模型只加载一次），49 帧冻结图逐帧过合体入口。

**分布（含 YOLO 交叉验证，即原规则）**

| board_presence | 帧数 |
| --- | ---: |
| absent | 33 |
| uncertain_vlm_no_yolo_yes | 10 |
| present | 4 |
| present_vlm_box_coarse_using_yolo | 1 |
| present_position_coarse | 1 |
| position_usable | 12 |

两帧真正有板的：一帧走 VLM 紧框（position 可用），另一帧 VLM 整幅框 → 退检测器框 → 仍过粗 → 位置不可用。

**问题**：那 10 个"不确定"里，**8 帧人工明确标为无板**——交叉验证在把正确的"无板"翻成"不确定"。

**改用"仅 VLM 判存在性"再算一遍（同一批帧的事后比较）**

| 规则 | present | absent | 不确定 | 真板 2 帧 | 明确无板却判 present |
| --- | ---: | ---: | ---: | --- | ---: |
| 含 YOLO 交叉验证 | 6 | 33 | **10** | 2/2 | 1 |
| **仅 VLM** | 6 | 43 | **0** | 2/2 | 1 |

**结论**：交叉验证在这个域里**误伤大于收益**，已把 `--cross-check` 默认改为 **off**；YOLO 案板框仍保留，但只在"VLM 框过粗时当位置候选"，不再参与存在性判断。

**如实说明**：这是**同一批开发帧上的事后比较**，不是独立测试；它足以支撑"关掉交叉验证"这个工程决定，但不足以宣称任何准确率。
