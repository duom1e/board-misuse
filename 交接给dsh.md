# 最新：板面标注流水线就位（2026-11-01）

"板在哪"只有 11/32 可用、几何法不可靠，唯一规模化路径是训板面检测/分割——缺的是标注。已建两端工具：`board_annotation_queue.py`（扫出有板帧、标记是否已有 `<图名>.board.json`、生成编号对照表）与 `build_board_seg_dataset.py`（人工框→YOLO bbox，按来源视频整组切分、SHA 去重、生成 data.yaml+manifest）。当前队列：**32 帧有板、31 帧待标**；数据集已能产出（现 1 条，远不够训练，manifest 里写明 val 需≥3 帧留出源才开训）。单张标注约 10–20 秒，32 张约 10 分钟。限制：矩形框非掩膜、样本仅 6 场景、标注仍是单人名目检。详见 板面标注流水线报告.md。

---

# 最新：人工框定板面可用（2026-11-01）

针对"板在哪"（32 帧有板里仅 11 帧模型框可用、几何法又不可靠），把最可靠的路做成工具：`set_board_box.py` 写人工框 + 预览，`review_combined.py --board-json/--board-box` 人工框优先（presence=present_human_box、source=human、position_usable=true，模型输出仅作交叉参考）。实跑黑色砧板帧：人工框定后位置可用，食物框因未达 80% 覆盖仍判 uncertain（护栏生效）。测试 44 项通过。限制：矩形框、不代表接触、不解决规模化，但可作板面分割/检测器的标注起点。详见 人工框定入口报告.md。

---

# 最新：几何法找板面不可靠（2026-11-01，负结果）

为补"板在哪"（32 帧有板里 VLM 框只有 11 帧可用），试了经典几何：边缘法（Canny+approxPolyDP 四边形）在 6 帧抽样上 **0 候选**；区域法（Otsu 双极性+minAreaRect）6 帧只有 1 帧框对（木砧板），其余锁到胡萝卜/鱼/背景，或板面占画面 >60% 被过滤。原因：砧板与食材/手/台面颜色亮度高度接近，灰度阈值分不出"板/非板"。结论：**位置问题不能靠廉价 CV 解决**，可行路径只有人工框定、或训练板面分割/检测器（需掩膜标注）。详见 几何法找板面_负结果报告.md。

---

# 最新：盲标更正后累计 32 帧有板全中、54 帧无板零误报（2026-11-01）

对 v1–v3 做"分歧帧逐帧回看"（v4 已做）：v1+v2 只有 1 处分歧，看放大图确认**我又错了**——E66 190s 是不锈钢台上的白色砧板，已更正。四次更正全部是"我漏看板"，模型四次全对。更正后累计（`experiments/cumulative_board_evidence.json`）：**有板 32/32 命中、无板 0/54 误报、有板帧中位置可用 11/32**；可用帧 86。旧结论同步更正：v2"误报 1/20"→**0/19**；单问入口"2/2、1/39"→**3/3、0/38**。方法学结论：单人目检可靠上限约 85–90%，凡分歧必看图为证，误报率一律视为上限。详见 盲标更正与累计证据报告.md。

---

# 最新：v4 多场景砧板 12/12 命中，并公开更正 3 处盲标错误（2026-11-01）

新素材：BV1bqsfzbELh（沉浸式切菜合集，多种厨房/木·黑·绿砧板）+ BV1xW411f7zP（KFC 油炸线，无板）。先冻结、盲标、后跑合体入口。**初判有 3 个"误报"，逐帧看图后发现是我把三块深色砧板误标成锅/托盘——模型三次都判对。** 已保留原始盲标并追加 revisions 与证据路径。修正后：有板 **12/12** 命中、无板 **0/8** 误报、VLM 解析 0 失败；位置可用仅 4/12。v3+v4 合计：**29 帧有板全部命中、16 帧无板零误报**。教训：单人目检标签噪声约 12%，凡"模型与标签不一致"必须逐帧回看。详见 新整源v4_多场景砧板与标签更正报告.md。

---

# 最新：新整源 v3 —— 17 帧有板全部命中、0 误报（2026-11-01）

找到真正的砧板作业素材（酒店砧板岗位 BV12a41187Ja、冷盘切配 BV1pq4y147jk），先冻结抽样与帧 SHA、再盲标、后跑合体入口：26 帧中有板 17、无板 8、看不清 1。结果 **有板 17/17 命中、无板 0/8 误报、VLM 解析 0 失败**；但 **位置可用只有 5/17**（特写镜头板面占满画面，框判过粗）。限制：17 帧里 14 帧是同一静止场景，有效场景仅 2 个。详见 新整源v3_砧板岗位对照报告.md。

---

# 最新：接第30次的棒——整体人体屏蔽不可行（2026-11-01）

第30次（food_human_negative10_v1，19:55 完成）结论：旧 mAP50 0.5915→0.6090，但鸡肉 4/6→2/6、食材检出 16/19→13/19、人体区域 3/4→1/4，误检降但召回退，不替换默认；报告自述"24 人脸仍 0.87 误肉、左手持刀仍出肉框"。

据此我做了上游 person 屏蔽探针（判据先写死，4 张人体区域图，三个食物模型）：人体区域误检被 person 框覆盖 **100%**（主判据通过），但非人体区食物框被连带屏蔽 —— 第28次 2/3、第30次 3/4（护栏 ≤10% 未过）。原因很直接：**厨师手里拿的食物本来就落在人体框里**。结论：整体人体屏蔽不可用；下一步应只屏蔽脸/头肩区域，并扩到 ≥20 张图另锁判据。详见 接棒第30次_上游人体屏蔽探针报告.md。

---

# 最新：粗框不再被当作位置证据（2026-11-01）

`food_region_gate.locate()` 新增 `board_position_usable`（默认 True）：板框被判"过粗"时，即使食物框 80% 落在板框内，也返回 `uncertain / board_box_too_coarse`，而不是 `board_candidate`；容器优先逻辑不受影响。`review_board_regions.py` 已接线（输出 board_position / position_usable / position_warning）。端到端对照：整幅板框 → uncertain，紧板框 → board_candidate。顺带修了输入健壮性：校验器现在剥 UTF-8 BOM。测试 34 项全过。详见 粗框不再当位置证据报告.md。

---

# 最新：框质量门槛落地（2026-11-01）

`review_board_only.py` 现在区分"看见板"和"框能用"：新增 `box_quality()`（面积阈值 0.6），输出 `board_position / position_usable / next_action`；整幅框改画橙色虚线并标注 "too coarse - frame it manually"，报告明确写"不要据此判断食材在板上"。49 帧回放：候选 6、位置可用 4、框过粗 2、未报出 43、unknown 0；有案板状态判断不受影响，只是把"看得见但用不了"显式化。测试 18 项（本模块）/32 项（含关联）通过。详见 案板单问入口报告.md 第九节。

---

# 第30次已核对完成

10轮训练和自动评估已完成，无后台。人体固定区域3/4→1/4，但鸡肉4/6→2/6、食材16/19→13/19，新厨房仍2/4，旧mAP .5915→.6090。36对叠图已看：部分手误检消失但别处手误框仍在，脸24仍误肉，部分鸡肉漏检加重。默认未替换。训练负例前后预测保存human_negative_replay_v1。报告补充目视结论；第28次仍为候选比较基线。下一步需平衡困难正负样本，不能只增负例或把区域计数当全视频误检率。

---

# 第30次训练与自动评估完成

food_human_negative10_v1/报告.md已比较食材、厨房、人体回归。fullframe_human_negative_comparison_v1已生成36帧对照，需目视复核。默认未变。不要重复训练，先读报告和人体结果；总表30次。

---

# 第30次自动评估已接入

finalize_human_negative_training.py正等待food_human_negative10_v1完成，状态human_negative_finalizer_status.json、日志finalize_human_negative_training.log，会话90766。结束自动固定组框/新厨房/36全画面对照/4人体区域回归，生成比较报告并更新总表与交接。不要重复排队；后续仍需看36帧叠图确认类型错误，标记计数不是全视频误检率。默认不变，当前训练继续。

---

# 进行中：第30次人体负例继续训练

food_human_negative10_v1，从28次last继续10轮，AdamW .0001/warmup0，多尺度；原4642条目+6张来自其他3视频的已看全图/手裁剪各5次，共4672条目/4640独特图。human_negative_replay_v1有review状态和SHA；audit_training_inputs增加显式extra_reviewed入口，本次调用。四个人体回归区域不入训练。结束需固定食材评估+人体回归+36全画面对照；默认不变。脚本train_food_human_negative10.py，同名.log，先查进程不重复启动。

---

# 已自行接棒：第29次结果已核对

第29次已恢复完成45轮，不要再启动/恢复，当前无训练后台。补玉米同标签mAP旧64 .5807→.6286、鱼果 .7444→.7539；但鸡肉4→3/6、新厨房2→1/4、食材16→11/19。补做human_region_corn_check_v1：人体区域误检3→4/4。因此不推广；默认未改，28次仍可作候选基线但有误检。原主日志停第8轮，恢复日志另存，status resumed=true；完整CSV已核对。报告补充恢复限制与人体回归。下一步优先完整画面人体混淆和独立来源数据，避免单纯加轮数。代码报告待GitHub同步。

---

# 最新：案板单问入口已落地（2026-11-01）

新增 `review_board_only.py`：一次推理只问"有没有切菜板"，只取 boards 段，同一严格校验器，输出仍 needs_review。49 帧冻结图回放：**可解析 49/49、unknown 0、有板命中 2/2、无板误报 1/39**。对比：基线一次问三件事 87.8%、拆三问合并 77.6%、强契约 85.7% —— 单问把格式问题清零。结论：案板判断可先独立成辅助复核路径，食材/容器作为可选附加信息。限制：有板样本仅 2 帧、本次为回放、开发集已被反复消费。详见 案板单问入口报告.md。

---

# 最新：拆成单任务提问后，案板判断变准了（2026-11-01）

把"板/容器/食物"拆成三次单任务提问，在同一批 49 帧上跑：**单独问"板"时解析 49/49，两帧有板全部命中（基线 0/2），无板误报 1/39**；但"整帧三段都成功"的有效率只有 77.6%（容器 45/49、食物 40/49 拖累）。结论：整帧一刀切有效率掩盖了"板判断其实已可用"。据此给校验器加了可选开关 `allow_partial_fields`（默认关闭）：只有 boards 段合法时可给 board_state，但整体仍 valid=false 并标 partial_fields/invalid_fields；boards 段本身有问题仍 unknown。测试 17 项（本模块）/31 项（含关联）通过。判据主项（整体≥95%）不满足，不采纳"整帧"口径；建议把"板判断"独立成产品路径。详见 方案B拆单任务提问报告.md。

---

# 最新：强契约提示词也压不住 schema 失败（2026-11-01，判据不成立）

把输出契约写死（只输出一个 JSON、给示例、每类最多 3 框），在同样 49 帧冻结帧上跑：有效率 87.8%→85.7%（更差）、v1 失败 1→2、v2 5→5，且 **board_state 改变 38.8%**（原正确沉默的洗菜机/传送带开始报有板）。两条判据全不满足。结论：**schema 失败是模型输出的内生行为**——它不遵守数量约束，加 token（300→600 已实测无效）也没用，改提示词还会牺牲稳定性。下一步建议走"拆成单任务提问"（每问只要 1–3 框），或产品侧同意"接受超量 + 明确标记 clipped"；两者都要另锁计划。详见 提示词契约实验v1报告.md。

---

# 最新：截断不是主因，超量生成才是（2026-11-01）

新整源 v2（两段未用过的备菜/菜场视频，25 帧，先冻结先盲标）跑 4B：无板 19/20 正确沉默、**误报 1/20**（推翻绝不误报的乐观说法），schema 失败 5/25。随后做了锁判据的**截断探针**：max_new_tokens 300→600，失败数 v1 1→1、v2 5→5，board_state 零变化 → **判据不成立，截断不是主因**。错误码细看：truncated 少 1、box_count 多 1，说明模型是**输出超量**（>6 框），给更多 token 只会更超。v2 唯一有板帧恰好 schema 失败，所以漏检应记为未知。

下一步最卡产品的点是**提示词/输出契约**（限框数、只输出 JSON），需新锁计划；有板样本合计仅 2 帧，仍需砧板备餐线新视频。详见 新整源对照v2与截断探针报告.md。

---

# 最新：新整源对照 v1 完成（2026-11-01）

24 帧（两段未用过的新中央厨房视频，抽样时刻与帧 SHA 先冻结、盲标先落盘）跑本地 Qwen3-VL-4B：**无板 19/19 零误报**，唯一 1 帧明确有案板的画面**漏检**（模型把备餐台判成 containers）。另发现 1 条 schema 失败其实是 **max_new_tokens=300 截断**，已新增独立错误码 json_truncated 并补测试（本模块 15 项通过）。限制：有板样本仅 1 帧、素材是加工/包装线、单人次目检、不是准确率。详见 新整源对照结果_v1.md。

下一步：需要砧板备餐线的新完整视频才能测召回；若调 max_new_tokens 必须另锁计划。

---

# 最新：第29次补玉米训练完成（2026-11-01，adopted=false）

45 轮已跑完（中间被会话中断 5 次，最终用 DETACHED_PROCESS+CREATE_BREAKAWAY_FROM_JOB+pythonw 跑通，不加 CREATE_NO_WINDOW，否则会新建隐藏控制台仍收到 CTRL_CLOSE）。结果双口径：含玉米标签 old64 0.5807→0.6286、鱼果 0.7444→0.7539（变好）；旧标签 旧mAP50 0.5915→0.5582、食材检出 16/19→11/19、鸡肉 4→3、新厨房 2→1（变差）；背景误帧两边 0/23。收益不覆盖损失，**不采纳为默认**，注册表未改。详见 第29次补玉米训练结果报告.md，数据在 food_fullmask_corn_multiscale45_v1/。

GPU 现已空闲。下一步优先：experiments/new_source_video_v1（24 帧已冻结 + 盲标已写完，标签在模型推理前落盘），跑 evaluate_new_source_v1.py 做新整源对照；以及把 29 次的双口径结论写进总表。

---

# 最新：第29次扩大数据训练及评估完成

food_fullmask_corn_multiscale45_v1/报告.md已生成与第28次的固定比较，corn_fullmask_fit_comparison_v1为640/960训练组框对照，总表已更新。请先读结果判断，不重复训练；默认未切换，代码报告待上传GitHub。

---

# 最新：区域输出校验已升级为可诊断 schema v2（2026-11-01）

region_response_validation.py 升到 schema_version 2：新增 box_nested（嵌套坐标）、box_extent_zero（零面积）、box_inverted（反向框）三个独立错误码，错误消息带精确 JSON 路径；结果新增 errors_by_code、field_counts、raw_length、raw_sha256。真实失败案例 experiments/qwen4b_temporal_presence_v1（prep_to_stove_010，food[2]=[[...]]）现在精确报 box_nested at $.food[2]。行为刻意不变：失败仍是 unknown/needs_review、不展开嵌套、不删框、不改历史指标。测试：本模块 14 项 + 关联 28 项全过。详见 区域输出校验改进报告.md。

同时：第29次补玉米训练被会话中断两次（停在 epoch 8/45，finalizer 卡在 waiting_training），已改用 Windows 计划任务 board-misuse-corn-resume 续训（与进程树解耦），当前在 epoch 9/45 继续；完成后再跑 finalize_corn_fullmask_training.py。

---

# 最新：人体区域固定回归已建立

human_region_regression_v1包含两张人脸/两张手部内部区域，原图及四张标注图已检查。完整画面候选中心落入人工区域则标记待复核；小数据模型2/4、扩大模型3/4（1脸2手），不能当独立准确率。不是训练数据，不用于自动屏蔽食材框。第29次完成后需在这些4张完整画面上重跑，并保留候选原始类别。报告与manifest已保存。第29次训练仍进行，原自动评估尚不含此新回归，需要完成后补做。

---

# 最新：全画面对照揭示人脸/手部误检

fullframe_multiscale_comparison_v1已比较27/28两模型36张完整画面并目视3页。扩大模型候选帧18→28/36，但人脸24、手部多帧误肉、洋葱25误肉，部分鸡肉/蔬菜漏检。背景裁剪0/23不能推全视频无误检，不推广默认。详情报告/index.html，原始predictions保留。第29次补玉米训练仍运行，自动评估队列正常。后续应补全画面人物/手部错误评估，不能把训练来源画面当独立测试。

---

# 进行中：第29次补玉米45轮

food_fullmask_corn_multiscale45_v1已启动：4648独特图/4656条目，通用yolo11n同起点/同多尺度45轮设置，VEG加入59。保留28次基线。训练结束自动生成旧及补玉米94图指标；finalize_corn_fullmask_training.py队列将跑组框/新厨房/训练尺度对照并生成含同标签对比报告，更新总表交接。状态corn_fullmask_finalizer_status.json，先查进程与日志不重复。新验证旧64/鱼果30同标签基线.5807/.7444。默认不变。

---

# 最新结论：第28次扩大数据有一致改善

训练与自动评估均完成，无后台。4634独特图/45轮多尺度：旧mAP .5915、鱼果.7474、鸡肉4/6、新厨房2/4、背景0/23、食材16/19。相对第27次均多项提升；640训练框10/14、96014/14，有尺度残余。补玉米统一标签后重评corn_label_baseline_comparison_v1：旧64 .5254→.5807、鱼果30 .5244→.7444，改善仍在。默认未替换，28次作为下一实验候选基线。下一版玉米数据4656条目/4648图与对应94验证已准备但未训练；应同标签同条件比较，不直接混用旧mAP。最新结果报告尚需上传GitHub。

---

# 最新：第28次扩大数据训练及评估完成

food_fullmask_multiscale45_v1/报告.md已生成与第27次的固定比较，fullmask_fit_comparison_v1为640/960训练组框对照，总表已更新。请先读结果判断，不重复训练；默认未切换，代码报告待上传GitHub。

---

# 下一版玉米标签已准备，未训练

food_fullmask_corn_replay_v1：4656条目/4648独特图（官方mask4584+本地72条目），VEG加入玉米59，保留原数据。audit_training_inputs通过，空标签26；原固定开发图片SHA排除。food_corn_validation_v1重建旧64+鱼果30共94张，6张含玉米，已看6张叠图；合并植物mask的组框可能含较多背景/遮挡，非完美实例框。新验证用于新范围，不能直接和旧标签mAP混比。第28次food_fullmask_multiscale45_v1继续运行，未改其数据、未启动下一训练。下一步先收28次结果，再决定新映射实验，若用新验证必须各对照模型重评。

---

# 训练并行审计：类别覆盖缺口

food_taxonomy_gap_audit_v1已核对数据卡类别与本地计数：玉米59未映射，343官方训练图出现，其中325已进入当前训练（因同图其他目标）；另有虾/贝/蟹、豆腐、菌菇未覆盖。不要把现两类说成全食材。下一版需固定覆盖规则并同步训练和验证标签，至少玉米符合当前广义植物食材定义；暂未改运行中数据或停训练，保留第28次对照。详情报告与counts.json。训练仍运行、自动评估队列待完成。

---

# 训练数据自动检查已完成

第28次仍运行。audit_training_inputs.py检查4642条目/4634独特SHA：无已隔离图、无固定验证/厨房/鸡肉开发图精确SHA泄漏；26张空标签均在目视复核允许清单。input_audit.json在food_fullmask_clean_replay_v1。已用已隔离图片做拒绝检查通过（training_input_guard_check_v1），并接入train_food_fullmask_multiscale45.py以后启动前检查。当前运行不重启，实际数据本次检查已通过。此检查不覆盖近重复或所有非空标注语义。自动评估队列仍等候训练完成。

---

# 自动评估已排队

第28次food_fullmask_multiscale45_v1仍运行。finalize_fullmask_training.py会等待完成（上限3h）后运行组框/新厨房/640和960训练拟合比较，生成正确新报告、更新总表与三份交接。状态fullmask_finalizer_status.json、日志finalize_fullmask_training.log，队列会话92832。不要重复排队。当前每轮约111秒，45轮约80多分钟，不是已完成。训练或队列终止时需检查状态与进程再恢复。默认模型不变。

---

# 最新进行中：第28次扩大官方训练标注

food_fullmask_clean_replay_v1构建完成4570张官方训练mask图+72本地条目，共4642条目/4634独特图；旧验证64及鱼果30的SHA排除。抽查8张转换框，非全量专家复核；原类别范围仍有遗漏非目标食物，不能声称全食材。第28次food_fullmask_multiscale45_v1：通用yolo11n起点，45轮多尺度约640—1280，数据大幅扩大，预计比前轮更久。脚本train_food_fullmask_multiscale45.py、同名.log，先查进程status防重复。保留原训练数据/权重，默认未改。完成需同固定组框/新厨房评估与640/960诊断，更新总表与报告。27次已完成且多尺度缓解训练尺度敏感，但新厨房仍0/4。

---

# 最新完成：第27次多尺度45轮

food_clean_multiscale45_v1完成，固定640：旧mAP .5240、鸡肉2/6、新厨房0/4、背景0/23、食材12/19参考类12/19。训练组框640从固定尺寸方案5/14→14/14，960两者均14/14；训练尺度敏感性缓解，但跨来源仍不足。默认未替换，无训练后台。报告已改正模板，总表27次。下一步扩大可靠来源覆盖，保留多尺度作为候选训练设置，不以训练拟合充当独立准确率。最近代码/报告尚需同步GitHub（最后远端仍较早）。

---

# 最新进行中：第27次多尺度45轮训练

food_clean_multiscale45_v1，脚本train_food_clean_multiscale45.py，会话54537。与通用45轮同数据623独特/631条目、同初始yolo11n.pt、同优化参数；新增multi_scale=1/3，约640—1280，算力随尺寸变化。AuditedTrainer记录observed_train_sizes.json，已观察含640、960、1280等20种输入尺寸，训练第3轮正常。先查日志/状态/进程不要重复。结束自动旧/鱼果/裁剪评估，还需report_food_clean_multiscale45.py（报告模板文字需重写）、evaluate_clean_multiscale45_kitchen.py、训练组框640/960对照，总表与交接并上传。默认不变。

---

# 最新：通用起点45轮完成，26次记录

food_clean_generic45_v1已完成，640开发退步：旧mAP.3608、鸡肉0/6、新厨房0/4。训练组框640合计5/14，960合计14/14，存在尺度敏感；新厨房960仍0/4。详见报告/scale_fit.json及generic45_kitchen960_v1。默认未改，无训练后台。下一步应考虑尺度变化训练与来源泛化，勿把960训练拟合当泛化改善。

---

# 最新进行中：通用权重重新训练45轮

food_clean_generic45_v1，脚本train_food_clean_generic45.py，日志同名.log，会话11297。通用yolo11n.pt80类起点（同n规格）转换两类，清理后623独特图/631条目，45轮960batch8，AdamW .001/lrf.1/warmup3，与此前训练设置不同，不能单归因初始化。状态文件training不代表进程仍活着，先查日志与进程，不重复启动。完成后旧/鱼果验证和42裁剪自动评估，还需组框和新厨房评估、加入总表、更新报告并上传。旧before仅复制历史foodseg参考，不是80类通用起点直接指标。默认不变。

---

# 最新审计：旧负样本生成入口

prepare_foodseg_v2.py以缺食材标注推断背景，已加main失败保护禁止再生成。历史数据未动。旧验证64张空标签0，但窄分类仍有范围限制；详情old_validation_label_audit_v1/报告.md。清理后模型仍从旧foodseg初始化，不能保证消除历史污染，下一步考虑干净通用预训练起训而非继续旧权重增量。当前无训练后台。

---

# 最新：清理后15+30轮已完成

两个阶段已评估并更新报告和训练总表，见food_empty_label_clean_training_v1及food_empty_label_clean_continue30_v1/报告.md。累计45轮，优化器在续训时重启。默认未替换。下一步读两个报告并汇总，不重复训练。

---

# 用户要求多训练：已安排累计45轮

当前food_empty_label_clean_training_v1原15轮运行中。队列queue_clean_continuation.py会在status completed后启动train_food_empty_label_continue30.py至food_empty_label_clean_continue30_v1，额外30轮，从前15轮last初始化，优化器重启，不等于不中断45轮。数据623独特图/631条目，默认不变。队列会话16732，先查python进程、两阶段status与queue_clean_continuation.log，禁止重复排队。队列30min超时则退出，不会在第一阶段失败时启动。保留两个阶段和相同开发评估；接下来待完成后跑组框、新厨房、拟合和总表更新。尚不能宣称45轮完成。

---

# 重要发现：空标签污染，2026-11-01

已目视67张空标签联系表，发现旧回放多张含葱/椒/水果/肉却为空标签。保守隔离41张含目标或存疑图片，不改原数据。empty_label_clean_replay_v1为623独特图/631条目。第24次food_empty_label_clean_training_v1已启动15轮960，从原foodseg初始化。先查同名.log和status避免重复；完成做固定开发评估、核查训练拟合并更新总表。空标签联系表与报告在empty_label_visual_audit_v1。不能把此前23轮全都说无效，也不能把全部退步归因于此，但应优先处理该实际监督错误。默认不变。

---

# 最新核查：2026-11-01

training_integrity_audit_v1已核查最近21/22/23轮训练：参数变化254/256、254/256、253/256；均15轮；无缺标签或非法数值；独特路径662/662/664。空标签65/65/67，包含有意负例，但所有空标签语义未经本次数值审计复核，清单empty_labels_for_review.json。脚本audit_training_integrity.py。不能以参数变化证明模型正确；默认不变，无训练后台。后续优先复核空标签是否误漏食材、标注语义和来源覆盖，而非继续反复小量训练。第22/23轮结果仍按各报告，不应推广。

---

# 最新：第23次训练完成

food_green_board_negative_training_v1已恢复完成15轮，固定评估已完成，无后台。旧mAP .5439、鸡肉3/6、新厨房1/4、背景2/23。整鸡图错误蔬菜两框仍在，肉框消失；局部负裁剪无修复证据，不推广。详见实验报告和chicken_predictions.json。总表23次，默认不变。

---

# 进行中：第23次训练

food_green_board_negative_training_v1已启动，672条目/664独特图/15轮960，沿用上一轮两张剁鸡图各5次曝光，增加两个已看图的负裁剪（刀面、发绿案板，不含肉菜）。数据green_board_negative_replay_v1，脚本train_food_green_board_negative.py，日志同实验名.log。先查状态和进程不要重复。完成需固定开发评估、剁鸡训练叠图对照、总表；默认不变。chicken_box_granularity_audit_v1已看图：剩余漏GT为小块皮脂，整鸡同时存在错误蔬菜框，不宜仅归因组框。

---

# 最新完成：第22次训练

food_chopping_exposure5_training_v1完成662独特图/670条目/15轮960，无后台。剁鸡图重复曝光5次，训练匹配1/3→2/3，但鸡肉开发4/6→2/6、新厨房1/4→0/4、旧mAP50 .5401→.5300、背景1/23→0/23。默认不变，不推荐加重复曝光。拟合对照operation_fit_audit_v2已完成；训练报告和两项固定开发评估已完成，总表22次。下一步检查组框粒度与更多独立来源，不要继续把相邻图或重复曝光当扩充来源。

---

# 进行中：剁鸡曝光次数对照

新诊断operation_fit_audit_v1：第21轮新增剁鸡训练组框仍1/3，几何亦1/3；此前肉操作5/5，第一视角菜5/6。新实验food_chopping_exposure5_training_v1将两张剁鸡图各重复曝光5次，662独特图片、670训练条目、15轮960，其余沿用21轮配置。从原foodseg初始化。不是新数据扩充，总步数略增，不是严格等算力。先查同名.log/status和python进程，勿重复启动。完成需评估旧开发、训练拟合、新厨房并更新总表；默认不变。

---

# 最新完成：第21次检测器训练

food_chopping_chicken_training_v1已完成662图15轮960，无训练后台。已运行两项评估，旧mAP50 .5401、鸡肉4/6、新厨房1/4、背景1/23、食材检出11/19、参考类11/19。对比660图有恢复，但未超过657图整体，默认不变。总表已加入21次。新增两张ROI三个肉组框只作训练；开发集仍保留。详细报告已修正旧模板文字，见该实验报告.md。下一步需扩大来源与可靠标签覆盖，不能把这些反复消费的开发集当盲测。

---

# 进行中：第21次训练

food_chopping_chicken_training_v1，662图15轮960。新增BV15J411D718帧1/6两个ROI三个肉组框，原图与overlay已检查，非专家；manifest为chopping_chicken_replay_v1，保留660图全部样本。启动脚本train_food_chopping_chicken.py，日志同实验名.log。先检查status与进程，禁止重复训练。完成运行report_food_chopping_chicken.py（检查模板文字）、evaluate_chopping_chicken_kitchen.py，加入总表；不自动改默认。阈值诊断完成但默认.25不变。

---

# 最新诊断：置信度与漏检

已运行diagnose_confidence_failures.py，输出confidence_failure_audit_v1。三模型固定阈值.01/.05/.1/.25，厨房4ROI和背景23裁剪，未修改默认阈值。660图模型.25类别0/4背景2/23，.01类别2/4几何3/4背景10/23；657图15轮.01类别3/4背景10/23。说明部分候选低置信且有类别混淆，单纯调低阈值代价很大，不宣称解决。该诊断未训练，无后台。第20次训练完成结论不变。下一步应补充更多来源的肉/鱼/蔬菜操作正例与易混手套刀具背景，保留现开发来源不入训练。

---

# 最新完成：2026-10-31 第20次检测器训练

food_pov_operation_training_v1完成660图15轮960，无训练后台。已运行report_food_pov_operation.py与evaluate_pov_operation_kitchen.py。657图15轮→本次：旧mAP50 .5485→.5403、鸡肉4/6→2/6、新厨房1/4→0/4、背景1/23→2/23、参考类别10/19→12/19。新增3张第一视角6组框未带来整体收益，不替换默认。详细报告在该实验及pov_operation_kitchen_validation_v1下。接下来不能把反复使用的开发集当独立测试；应扩大来源覆盖和标注质量，而非仅继续添加少量相邻帧。

---

# 进行中：2026-10-31 第一视角操作增量

food_pov_operation_training_v1 已启动660图15轮960训练，进程会话56647。不得重复启动；先查status.json和food_pov_operation_training_v1.log。新数据pov_operation_replay_v1：BV1dHckzwEBA的930/940/1000秒3张ROI、6个蔬菜组框，已看原图与overlay，非专家标注，单来源相关画面，只训练。完成后运行report_food_pov_operation.py、evaluate_pov_operation_kitchen.py；将该实验加入build_detector_training_ledger.py后重建总表。注意报告旧模板的描述需核对更新。默认模型未变，尚无新训练结论。上次19次训练完成结果仍保留。

---

# 最新接力：2026-10-31

第19次检测器真实训练已完成：food_meat_operation_30epochs_v1，同657图30轮。上轮中断使进程终止，使用resume_food_meat_operation_30epochs.py从28轮检查点完成，results.csv核对30行。恢复随机状态未经逐步复现核验。默认模型未变，无训练后台。

15→30轮：旧mAP50 .5485→.4899；鸡肉4/6→3/6；新厨房1/4→1/4；背景1/23→0/23；食材13/19→12/19，参考类10/19→11/19。整体无收益，不推荐继续仅加轮数。详情food_meat_operation_30epochs_v1/报告.md和meat_operation_30epochs_kitchen_v1/报告.md。检测器总表已核对19次，未微调大模型。下一步增加不同厨房操作来源的可靠训练标注，现有开发样本不入训练。

---

> 2026-10-30操作肉域补充真实训练完成：prepare_meat_operation_replay.pyBV1dK4y1M7Qe12帧筛4ROI5框全看(低清/第一截断)，合统一653→657，train_food_meat_operation.py15轮960，验收640。food_meat_operation_training_v1原鸡held4/6，food13/19类10，bg1/23，旧map50 .5485/.95 .4031，新鱼果val .6569/.5350；新厨房4GT终于1/4(02容器肉)，其余鱼/菜/肉仍漏，对比上轮0；不是独立统计显著收益，域间退步不采用。新厨房与新valSHA无交，ledger核18轮，无后台默认改变VLM微调。

> 2026-10-30类别统一真实训练完成：merge_aligned_food_replay.py旧465fs mask全重建含鱼果，559原+120扩充-26重=653独特，保留47非mask旧及增量，官方train/新val30 SHA无交。train_food_taxonomy_aligned.py15轮960，report原鸡held4/6，food13/19类12，bg3/23，旧窄64map50 .5505/.95 .4185；新扩展30val .6353→.7081/.95 .5342→.5707。evaluate_taxonomy_kitchen_source.py新厨房4GT仍0/4且无预测，拒绝采用。数据/标签/样本量同时变非单因素；16新+8旧叠图样例已看非专家全量。实验food_taxonomy_aligned_training_v1与taxonomy_kitchen_validation_v1，ledger核17轮，无后台默认变化VLM微调。

> 2026-10-30重要类别覆盖纠正：原FoodSeg meat46/47/48/51+49/50不含fish54，veg69..96不含fruit25..45，与VLM口径不一致。prepare_food_taxonomy_alignment.py原mask转换150图(官方train鱼60果60/val15+15)，新classes含鱼与水果，16样例叠框全看，新splitSHA无交，integrity记录与旧512回放重复。food_taxonomy_alignment_v1报告/manifest/yaml/labels，尚未训练。下步必须统一旧回放全部mask标签并去重，不能同图旧窄标签+新宽标签混训；鱼0/4解释需加覆盖限制。无后台默认变化。

> 2026-10-30扩充固定来源验收kitchen_source_validation_v1：BV1tP4y1R7Ye02/03/04/05人工4限定ROI(3肉含鱼头1菜)4粗组框叠图全看，先锁后对照；06/07姜肉混合预先排除。prepare_kitchen_source_validation.py/evaluate_kitchen_source_validation.py 5模型固定.25/640rectFalse/.5IoU：original/bamboo/carrot几何1/4类0，cleaning960几何0类0，multiveg几何0类0。历史查看源非盲测，手工ROI与粗GT限制，无全图precision；这4留在固定开发验收不转训练。来源/图SHA/原预测/报告存，未训练默认切换后台。结果说明16轮旧鸡2图不代表厨房泛化，下步必须新验收约束训练。

> 2026-10-30多蔬菜补框真实训练完成：prepare_multiveg_group_expansion.pyBV15gZzYXE4N01/03/05/07四整图14框全看(非专家含密集遮挡)，food_group_boxes_v6。train_food_multiveg960.py559图15轮960/验收640，同64val鸡源隔离。food_multiveg960_v1鸡held2/6(前3)，bg3/23(前2)，food13/19类10，旧mAP50 .5549/.95 .4179，拒绝采用。总表核16轮，无后台默认改变/VLM微调。下一步优先统一粗组框/单物体粒度和扩大来源验收，避免继续少量增量混训波动。

> 2026-10-30清洗背景960真实训练完成：train_food_cleaning_negative960.py555图15轮，新增BV16g9nYNE6d02/07为肉菜两类空背景(乳白液体不是无食物，绿擦洗04/05不明排除)。固定640验收，food_cleaning_negative960_v1鸡held3/6(前9604)，bg2/23(前4)，旧mAP50 .5801/.95 .4425，food10/19类9；仍取舍不采用。记录cleaning_records/last/报告，ledger核15轮，无后台默认改变VLM微调。

> 2026-10-30真实960训练完成train_food_resolution960.py553图15轮，与640基准训练图SHA与关键配置同，controlled_settings存；所有验收仍640rectFalse。food_resolution960_v1鸡held4/6(前3)，旧map50 .5618/.95 .4290，bg4/23(前0)，food10/19类10，训练旧9框4，误报退步拒绝采用。训练总表核14轮，权重/原结果存，无后台默认切换VLM微调。

> 2026-10-30蔬菜增量真实训练完成：prepare_carrot_group_expansion.py胡萝卜12帧筛3图4框全看，food_group_boxes_v5。train_food_carrot_balance.py基于550竹板配方+3veg=553，15轮全层同64val，鸡留出未训。food_carrot_balance_v1鸡held3/6(竹板4)，旧map50 .5799/.95 .4338，bg0/23 food11/19类8；比竹板恢复旧域food但held少1，候选不采用。训练总表核13轮，无后台默认改变VLM微调。

> 2026-10-30冻结真实训练完成train_food_frozen_backbone.py1062条15轮freeze10，同64val/关键配置；audit_food_frozen_weights核99前部参数张量不变、153后部更新(不含BNbuffers)。food_frozen_backbone_v1鸡held1/6(全层2)，旧mAP50 .5197/.95 .3924，bg0/23 food15/19类12；crop改善旧域退步拒绝采用。检测器真实训练总表核12轮，参数审计/last完整，无后台默认变化VLM微调。

> 2026-10-30真实1024旧图回放对照完成train_food_bamboo_replay1024.py1062条15轮；原512全保留，新增组框/正负/64val同，replay_integrity。food_bamboo_replay1024_v1旧mAP50 .5875/.95 .4227，bg0/23 food13/19类7，鸡held2/6(512竹板4)，训练旧9框8/未匹配预测11，恢复旧域但鸡收益回落，不采用。build_detector_training_ledger.py核11轮results.csv15+lastSHA+oldafter，检测器真实训练总表.md。无后台默认变化VLM微调。

> 2026-10-29因用户要求转回真实检测训练：train_food_hard_negatives.py 128旧+16紧crop负144，15轮AdamW.0001全层固定last；train_food_balanced_detector.py同配置+11已有肉末检测框正155，两个真实训练完成。64相同旧val/trainSHA无交。原/负/平衡：训练背景FP2→0→0/16，训练外源背景FP2→0→0/7，19foodcrop有候选11→7→11，出现参考类8→6→10；旧mAP50 .5777→.5556→.5414。正负平衡开发改善但旧域退步，默认未改，无VLM微调后台。两个实验目录before/after/plan/last.pt完整，食材检测两轮训练对照.md与compare_food_detector_training.py。后续扩大检测框正例与旧域回放，不只加负。先前视频辣椒5点pipeline完成但未审计，不冒称效果。

> 2026-10-29增加已消费BV1XN411q7XJ12帧筛查，2原图6crop全看(3veg3background)，food_presence_patches_v3共5源42。固定原120/Adam.01/seed/冻结YOLO640rectFalse做5折整源训练food_presence_head_v2：鸡6/8辣椒7/10猪7/8新增椒4/6王8/10。同旧36=28/36→28/36无净改善，新增6单列不混分母；刀修但肉类退步，拒绝采用。comparison.json核源视频/原帧/cropSHA隔离与默认SHA，报告.md完整；训练结束无后台。run_food_presence_expansion_v2.py准备数据，train_food_presence_head_v2.py可复现训练，report_food_presence_expansion.py对照。无VLM微调，勿再调这42追分。

> 2026-10-29完成跨源食材背景真实小头训练：新增已有BV14W41197Rd/BV1264y1F7a6各12帧筛查，4原图16精确crop全看；food_presence_patches_v2合旧36/4视频/8原图=20背景10肉6菜，非专家开发。冻结原YOLO11s640rectFalse CPU嵌入L2，4折整视频留出训练Linear3 Adam.01固定120seed20261029最后轮；各train全拟合，留出鸡6/8辣椒6/10猪7/8王9/10。背景19/20肉7/10菜2/6，8错例全看（鸡皮误菜/辣椒蒜误肉/刀误菜/小肥肉漏），拒绝自动采用不调参追分。food_presence_head_v1含锁计划/4权重/features/results/audit/errors.jpg/index.html/报告.md；源视频原图crop跨折SHA无交集且默认SHA未变，无VLM/主干微调/后台。非盲测或自动定位准确率。脚本expand_food_presence_patches.py train_food_presence_head.py report_food_presence_head.py。后续补不同源黄色皮脂与红绿菜/刀背景，当前36固定开发回归。

> 2026-10-29手刀食材成对数据完成：screen_kitchen_sources.py新增--prefix/缺info保留URL，已有BV14K4y1x7LF、BV1Ys411L7r6各12帧筛查；4原图20手工精确裁剪全看(11background6veg3meat)，food_presence_patches_v1/reviewed_manifest.json非专家候选仅本地分类训练资格，detection资格false；classification_candidates按源/类导出SHA全核，不分train/test。固定YOLO .25/640/rectFalse CPU背景误1/11(纯手meat)，食材有候选3/9；原4B精确crop无margin 11unknown+3meat6veg符合参考。不是独立准确率/自动定位，不能推翻上批手刀误meat；肉仅一个来源不可现在三分类训练验收。3脚本prepare/evaluate/export，index.html与报告.md，编译过、无训练/默认切换/后台，未浏览器验收。下一步不同来源肉类与手遮配对，勿随机拆同源或把背景crop原图标空。

> 2026-10-29完整候选复核完成：review_food_candidates_batch.py 固定36帧全部27YOLO框+16位置排除框=43，先看4页裁剪锁非专家参考37可比6不确定，再原提示/10%margin/512/12token冻结4B一次。YOLO可比19/21与参考一致：手刀00仍误meat、鸡肉23误mixed；骨架08/水果31意见符合参考，不能自动覆盖。16位置排除=10meat5veg1unknown，与参考16一致但不恢复板上归属。43原框/源crop SHA核验，32软件测试通过，无训练/默认切换/后台。food_candidate_review_v1/index.html与报告.md，未浏览器验收；已消费非盲测不报准确率，training_eligible全false。下一步跨源手刀负例与小食材正例，保留本批固定回归，不自动伪标训练。

> 2026-10-29完成完整36帧双路线批次：run_food_batch36.py，food_batch36_v1首图保存变量错失败保留，修复后v2原输入prompt完整36(两鸡视频+菜教学各12)。4B35有效1失败，29板帧57food候选=41板(29meat12veg)+7容器+9uncertain；3场景页+4页41裁剪全看，默认全needs_review。原YOLOCPU同36实跑：鸡1肉8菜2/8帧检出，鸡2肉4/4帧，菜肉2菜11/6帧；4可疑框裁剪已看主要手/刀、鸡骨架误菜、手部误肉，yolo_suspects.json不自动GT。IoU.5双方可分配分歧0不代表一致(框不同)，82软件测试过，无训练采用后台。完整交付v2/evidence/index.html与批次报告.md，10问题待复核；下步批量标注手/骨架/小肉/容器，勿只清楚肉菜或报41/41。

> 2026-10-29食材定位/类别分离核查完成：audit_food_localization.py，food_localization_audit_v1已消费4图人工7粗GT(肉4菜3)，原图/叠框已看，manifest SHA/范围非穷尽/训练资格false；旧自动food_regions IoU.5一对一匹配5/7，5匹配类别2肉3菜一致，2未匹配手上小肉/边缘碗肉（不全称完全漏看）；无全图FP资格。白菜手持仍板候选不证接触。非盲标无训练新推理/采用后台，报告.md。下一步多源食材定位监督，分开位置类别接触，不只报meat输出数。

> 2026-10-29蔬菜视频整段入口完成：vegetable_video_full_v1 BV15gZzYXE4N0—310/30s11点(实际0..300余10.4s未采)，10结构有效1约240s未知、6板候选；11分类裁剪6veg5unknown0meat，原图/全部裁剪已看，0秒4小白块/119.987手遮unknown不补分，29.997板背景漏、89.99手持菜可能2D板候选不证接触。audit SHA通过，证据页含视频时间线，未浏览器验收，无训练采用后台。核查报告.md；非新源盲测，不宣称肉菜混淆已解决。

> 2026-10-29可移植模型路径完成：region_vlm_runtime.resolve_local_model支持BOARD_VLM_MODEL_DIR，无metadata时明确提示；run_video_evidence.py --model-dir传子进程并计划记目录/config SHA/revision null。未指定保留原默认；坏override拒绝不回退，无自动下载升级。真实explicit_model_path_smoke_v1 49秒1点成功候选板，57测试过，无训练/采用/后台。说明一键视频识别说明.md末节；非跨机一键安装，仅消除本机实验metadata依赖。

> 2026-10-29身份复核接入视频入口：board_identity_review.py + review_board_video.py --board-identity-review默认关，replay拒绝，复用同4B逐自动板框15%crop/12token保留原文/原框，无删框否决、不影响food_regions；未来中文report含身份段。真实49/74两点board_identity_entrypoint_v1跑通，意见数等原板框且detection_removed全false检查；报告本轮离线补身份段。新增3测试总55通过，无训练/采用/后台。仅工程联调非新准确率；other/unknown仍人工复核。

> 2026-10-29新来源身份任务完成：下载BV13s411H7dn207.32s+BV1TS4y1K7ZH330.2s，boardcare_sources_v1各12抽样已看、4原图人工ROI先锁(手持圆/刀遮圆/白圆/手遮方)，probe_board_identity_newsource.py沿用旧身份prompt15%/512/12，4全board，裁剪已看。board_identity_newsource_v1结果保留，非自动定位/全图GT、多板ROI只一个，字幕可见影响解释；不能推翻旧8难例误伤风险，不自动过滤，无训练采用/后台。报告新来源案板身份复核报告.md。

> 2026-10-29候选身份困难真板验证完成：probe_board_identity_hard.py，board_identity_hard_v1锁同视频8人工板ROI(6手遮2边缘)，沿用原prompt/15%/512/12token，5board2unknown1other(510s)，8裁剪已看。按非board删会误伤3/8，unknown保留仅删other也误伤1/8；不接自动否决，原框全保留，无训练/采用/后台。人工ROI不是自动找回漏检，不是新源盲测。报告候选复核困难真板验证报告.md，勿只引用上一轮定向6真6假好结果。

> 2026-10-29YOLO候选4B身份复核定向探针完成：review_detector_candidates.py，detector_candidate_vlm_v1已有已知结果选6真匹配板+6无板误框，每原图1候选锁计划，15%margin/512/12token固定prompt一次，6真全cutting_board，6假2table_surface3food_container1other。12裁剪联系图已全看。非盲测/泛化12/12，真板未覆盖漏检难例，other不能自动删框；所有原框保留needs_review/detection_removed=false，无训练采用/后台。报告候选案板大模型复核报告.md。下一步可选二次意见+新源困难真板误伤验证，不直接过滤。

> 2026-10-29完整场景混合真实训练完成：run_broad_board_trial.py，broad_board_trial_v1旧OI+已有LVIS train去重后526+24视频=550独特/408框/190负，固定10轮AdamW.0003全层，排除OI/LVIS valtest SHA+圆板留出SHA无重合。OI mAP50 .9117→.9145但.95 .7404→.7265；已消费LVIS57 val mAP50 .1512→.3085/.95 .1027→.2028，未看LVIS test；视频训练正3/12→5/12、负误3/12→2/12，圆板留出6/6→3/6，拒绝采用。候选叠框已查，merge_integrity/audit留存，默认不变无后台。报告完整场景混合训练报告.md；不是新增550素材，不重复加242，不称新盲测。下一步跨源圆板边缘手遮监督，勿再无目标加轮数。

> 2026-10-29标注与评估口径审计完成：board_annotation_audit_v1旧train242/214框近整幅29面积>=90%53，val10框近整幅5大7，test25框近整幅9大16；看新24训练叠图+旧12最大框，官方OI标签近整幅不等于错，有多板包络候选需复核未改。旧24图默认/260多来源实跑.25/.5/640/rectFalse一对一：大框15/16 vs14/16，其余5/9 vs8/9，FP7 vs2；非AP、较小面积不等于小目标，不能翻转此前新源失败。报告案板标注与评估构成审计报告.md；无训练/采用/后台。后续分层+跨源数据，别只盯旧mAP。

> 2026-10-29统一分组回归完成：evaluate_board_cohort.py，board_cohort_v1锁30图SHA去重/7视频/18正12负，5YOLO同.25/640/rectFalse/IoU.5实际150预测；按视频明确本项目训练暴露，非盲测；报告35分组含匹配/负例误帧/额外框，integrity核150唯一模型图对。原默认分层：手遮圆板0/6、边缘0/2、满屏板3/3、木桌圆板3/3、方板3/4、负例3/12误帧；不得把近景易例混为困难总准确率。无训练无采用无后台。下一步新来源手遮/边缘优先，当前30固定开发回归不再新测试。

> 2026-10-29负例增强真实训练完成：multisource_negative_trial_v1旧242+24(12正12负)=266固定10轮AdamW.0003全层，同上轮配置。新增6锅盘钢台原图已看；同12负例实跑上轮误5/12→本轮3/12(默认3/12)，训练正例9/12→10/12；训练外圆板6/6→4/6额外3框，旧mAP50 .9092/mAP50-95 .7226，拒绝采用。audit/候选叠框/annotation_provenance/previous_same_negative_set已存，默认SHA不变、留出SHA不重合，无训练后台（同负例评估session97333已输出5/12需如必要核结束）。报告多来源负例增强训练报告.md；下一步扩大标注源/分层，勿再反复小加图调参这批。

> 2026-10-28多来源YOLO真实训练完成：run_multisource_board_trial.py，multisource_board_trial_v1旧242+4视频18(12正6负)=260图，10轮AdamW .0003全层无mosaic/warmup固定last；2另圆板视频6图不训练且SHA无重合但此前消费非盲测。旧mAP50 .9117→.9461、mAP50-95 .7404→.7599；训练正例3/12→9/12，但训练负例误帧1/6→3/6；留出匹配6/6不变新增2额外框，继续拒绝采用。audit/候选留出叠框已查，10轮默认SHA不变无后台，无VLM微调。先读多来源案板训练报告.md。下一步更多跨来源易混负例而非只报mAP。人工复核控件此前已加但未浏览器验收，当前勿转UI。

> 2026-10-28原视频时间线接入完成：add_video_timeline.py验证pipeline_plan视频SHA后复制本地视频并加视频控件/10时间按钮/10卡片跳转；video_evidence_full_v1/evidence/index.html已更新。timeline_checks核副本SHA/20链接/重复安装不变通过，50规则测试仍过；未做浏览器点击验收不冒称播放已实测。run_video_evidence.py未来自动视频绑定阶段，失败写status；说明一键视频识别说明.md末节。无新增推理训练后台。

> 2026-10-28一键入口整段0—198s/20s验证完成，video_evidence_full_v1共10点0..180，10结构有效9板候选6点板肉意见，20/40/120几何未过；末18.56s未采样，不声称核验盛盘。联系图已看，audit原帧SHA/10叠框/10events皆needs_review无配置用途检查过。报告整段采样验证.md，页面evidence/index.html，pipeline完成无后台、无训练。180小框/unknown不能当实例准确率，后续时间序列证据展示勿合并伪违规事件。

> 2026-10-28交付一键视频到证据：run_video_evidence.py默认unconfigured仅观察，显式--demo-allowed才假设规则；含视频SHA/阶段日志/状态/失败退出，无覆写。真实BV15J411D718 24/49/74三点跑通，板3点、板肉意见0/1/3，均needs_review无错用判断。最终叠框页video_evidence_endtoend_v1/evidence_final/index.html，原evidence保留；build_misuse_demo.py已支持默认无规则+叠框原图。50测试通过，无训练模型切换/后台。先读一键视频识别说明.md。

> 2026-10-28按用户对冲突含义的反馈修正页面：build_misuse_demo.py及3已有HTML标题改案板食材与演示规则对照，疑似冲突可见标签改与演示假设不符，顶部注明未配置真实现场用途、不能认定视频人员错用；肉菜均可页说明相应调整。内部suspected_misuse兼容键未改；3规则测试过，无新增识别/训练。

> 2026-10-28演示证据v2已生成：misuse_demo_evidence_v2/index.html，6卡片真实时间从round_board_sources_v1/sources.json绑定原帧SHA核验，青板/红板食材候选/橙其他候选、板/容器覆盖比例、可展开原图，events.json补来源URL/帧号/视频SHA。5疑似采样+1不足仍只菜假设，不是真违规。输入round_food_chain_v1/timed_results.json，原结果未覆写；基础build_misuse_demo.py未改，v2为离线导出加工，无新增模型/训练。

> 2026-10-28交付端到端规则证据HTML：build_misuse_demo.py接受视频/链路results.json，显式假设所有板同用途，保留未知不判罚、不伪造置信度。misuse_demo_vegetable_v1/index.html只菜假设=5疑似采样+1不足，misuse_demo_both_v1肉菜均可=5无支持冲突+1不足，不是事件数/真实违规/合规；原图、源SHA、events.json可追溯，可筛选。3新规则测试总49通过；无训练模型变更。先读疑似错用证据演示说明.md，未来现场逐板配置/时间追踪仍需做。

> 2026-10-28位置原因可解释性完成：food_region_gate.py新增location_reason/geometry_only，规则阈值未变；round_food_chain_v1/location_reason_audit.json旧9候选重算location9/9不变。2uncertain板覆盖59.84%/71.01%<80%；1容器候选容器覆盖100%优先，不代表真接触。新增边界原因测试，46相关测试过。无训练无默认模型变更，后续入口JSON自动携原因；报告末节已更新。

> 2026-10-28圆板鸡肉自动全链完成：run_round_food_chain.py，round_food_chain_v1六帧结构有效，共9食材候选=6板候选全meat+2uncertain+1container unknown；5帧有板肉意见，03仅uncertain，03/04叠框已查。报告/results/原文/裁剪保留，无训练无采用无后台。不能将6肉意见说准确率或物理接触，未全量食材实例GT；下一步复核合框/重复/不确定位置。

> 2026-10-28两新圆板源验证完成：BV1hK411u73L 179.33s+BV15J411D718 198.56s，各12抽样+各01/04/08原图先人工锁6框，三YOLO固定.25/640/rectFalse IoU.5匹配默认6/6、普通3/6、保守5/6，额外框box_audit.json，原/普通叠框已查。round_board_sources_v1含sources/locked/results/报告.md，evaluate_roundsource_boards.py。近景板填画面IoU较易，部分框偏宽，不称精准/现场准确率；两实验继续拒绝，无训练无后台。下一步可做清楚鸡肉类别/候选关系诊断，已消费视频不可当盲测。

> 2026-10-28完成4B与三YOLO新源正负完整诊断：crosssource_vlm_v1锁原4正+4负，原提示4B512/300token一次跑，4正IoU匹配4/4，4负3有效无板+1JSON失败未知，7/8结构有效，不补分。4B料理02边界伸桌面虽IoU .791，不称精分割。补审YOLO正例额外框：默认匹配3+额外2，普通4+额外0，保守3+额外1；负例误框帧0/3/2。报告/summary/原文错误/叠框已看，未训练未采用，进程结束。先读crosssource_vlm_v1/报告.md；不只报命中率忽略多框，不把未知计无板，不证明食材接触。

> 2026-10-28跨新视频正例诊断完成：下载BV15gZzYXE4N蔬菜教学310.41s+BV1aRmLBmE7i料理1040.53s，各12抽样看图；人工原图先锁4板框（各2帧），三YOLO固定.25/640/rectFalse/IoU.5匹配默认3/4、普通4/4、保守3/4。叠框已查，结合普通旧回归下降+新负例3/4误框继续拒绝采用。kitchen_positive_sources_v1含sources/locked_boxes/results/报告.md，evaluate_crosssource_boards.py。无训练无后台；4帧已消费非未来盲测，非商业后厨准确率、作者场景隔离未知；继续补圆板困难数据，勿单报4/4。

> 2026-10-28新源负例验证完成：下载BV1eB4y1J7B6 312.93s与BV1Tj421f7ph 96.03s，各12抽样已查看，无适合圆板正例；前者酒店后厨/后者讲述。锁定酒店03/04/05/07四无明确板帧，三YOLO固定.25/640/rectFalse实跑，错误候选帧默认0/4、普通3/4、保守2/4，叠框已查。两实验继续拒绝采用，无新增训练/后台。kitchen_multisource_v1含sources/negative_plan/results/报告.md；筛选脚本screen_kitchen_sources.py，评估evaluate_new_kitchen_negatives.py。继续找跨来源切配正例，不将这些已消费负例作盲测。

> 2026-10-28第二轮保守YOLO训练/回归完成：run_cutting_board_conservative.py、kitchen_cutting_trial_conservative_v1；同252图固定10轮，freeze10/AdamW lr0=.0001 lrf1 warmup0 mosaic0，固定last。新训练8板匹配0/8，2负例误框1/2；旧24图mAP50 .9395、mAP50-95 .7412（原.9117/.7404），precision .8790较原.9325下降；旧能力保住但新场景未学会，仍拒绝采用。decision/候选叠框已查，10轮252路径无train-val/test交集、默认SHA不变，进程结束无后台。报告圆木案板保守训练对照报告.md；下一步不同来源困难数据，不再只调同8帧追分。

> 2026-10-28已完成真实YOLO继续训练与回归：run_cutting_board_trial.py，kitchen_cutting_trial_v1旧242+新10（8板2负）固定10轮/seed42/last，无大模型微调。IoU>=.5 conf>=.25训练图匹配旧0/8→新5/8，负例误框1/2→0/2；旧24图25目标mAP50 .9117→.8147，mAP50-95 .7404→.6213，回归失败拒绝采用。decision/results/新旧叠框已查；完整10轮、252图、train-val-test路径无交集、active SHA不变通过。报告圆木案板训练与回归报告.md；训练已结束无后台。新视频已是训练源不可当独立验证，下一步不同来源困难板数据+保守训练方案，不追加轮数追分。

> 2026-10-28重要更正+YOLO实际对照：逐张原图看420/600s无明确板，旧联系图GT错误，保留旧文件；manual_presence_v2为非盲修订8可见2无明确可见1未知。VLM TP6 FN2 FP0 TN2（原4漏检撤回，真实漏300/390）；默认YOLO固定.25/640/rectFalse TP2 FN6 FP1 TN1。compare_cutting_board_yolo.py及yolo_comparison.json/橙框联系图已查，权重SHA留存；无训练无采用。先读本批结果.md最后更正节，勿传播旧4漏检。

> 2026-10-28案板框补标完成：prepare_cutting_board_labels.py，kitchen_cutting_batch_v1/board_labels_v1含6原图人工框/YOLO txt/manifest SHA/截断/分组，overlay已逐图检查。既有6检出帧IoU .827—.934，仅条件定位诊断，不覆盖4漏点。未训练，不能只补清晰正例追分；下一步补边缘遮挡框及不同来源验证，保留食材未知。

> 2026-10-28完整后厨切配批次完成：kitchen_cutting_batch_v1细筛300—600s固定11点，推理前manual_presence标10部分可见/1未知。4B真实完整类别+上下文运行live已结束，6检出4边缘遮挡漏（300/390/420/600），480未知排除，无负例不可算精确率；330手附近误食材风险。presence_audit.json/本批结果.md及联系图已查。食材身份未确定，不编肉菜训练真值；本轮无训练无后台，不重跑追分。

> 2026-10-28第二批B站下载筛选完成：BV1dHckzwEBA 1044.3s低清101.53MiB，8帧看见圆木板切配，优先开发候选；BV1dC7H6PEoG 309.57s主要炒菜，保留非板/容器对照候选。原视频待导入视频/kitchennew_*，SHA索引/联系图在kitchen_search_20261028。未训练/许可未核实；下一步细看300—620s切配段标实例，勿称全片无剪辑或错用真值。下载均已完成无待轮询进程。

> 2026-10-28用户明确授权自行从B站找后厨视频。新增下载kitchennew_BV1dK4y1M7Qe及BV1EBR4BCEsy，各抽8帧已查看；前者肉刀木板近景开发候选，后者讲解用餐为主排除。来源/SHA/初筛说明在kitchen_search_20261028，未训练；继续找真实商业后厨连续切配。

> 2026-10-28已建立small_target_review_v1：6条可追溯补标候选（3小区域弃答+3预先不确定压力），queue.json保留来源/裁剪/意见，review.jpg原裁剪展示。全部pending_expert、training_eligible=false、许可未核实；无训练。build_small_target_queue.py可复现。已消费开发源不可转作独立测试，下一步补实例位置标注及来源隔离数据。

> 2026-10-28新来源上下文验证已完成并离线审计：context_new_source_v1锁定16裁剪/5帧，13计分10一致（肉1/2、菜4/4、混合1/3、非食物4/4），另3预先压力不计分。5含肉目标仅2保留肉证据，3小区域unknown；整图4次/1帧跳过，矛盾0，未恢复肉证据。混合整图无法核验局部身份，不能作为自动纠错；未训练未采用。manifest/原图/裁剪SHA审计通过。先读新来源上下文复核验证报告.md及context_new_source_v1/audit.json；旧crop联系图两熟肉小区域标签是初稿，推理前最终manifest已改mixed，勿误用。下一步小目标位置标注/来源隔离与许可核实，不重跑此批追分。

> 2026-10-27上下文复核已接通：food_context_review.py/视频--food-categories --food-context（默认关），保留局部标签/原文/位置，整图仅提供矛盾提示，同模型两任务非独立证据，不覆写不判罚。21条旧局部/9帧定向开发（7次整图、2帧全未知跳过）：4矛盾(2小胡萝卜误肉+1非食物误菜+1真菜被整图漏看)，9不矛盾5证据不足3未跑。不能称准确率提升。胡萝卜50s真实入口跑通2矛盾，肉标签仍保留；45测试+CLI依赖检查过，默认旧13回放12+1未知不变，历史SHA不变，无训练后台。先读食材局部与上下文矛盾复核报告.md；下一步新源小区域真肉真菜非食物验证复核负担，勿用整图菜覆盖局部肉。

> 2026-10-27类别阶段压力验证完成：2新导入视频+1已消费沙拉照片，17手工ROI先复核锁定，15计分+2预先不确定不计分。固定4B/原类别提示词一次运行，肉4/4菜3/3混合2/2非食物5/6；水槽/绿白抹布边误vegetable。2压力姜片unknown/姜排骨meat，不补分。仅人工ROI类别诊断，非自动定位/板上接触/现场14/15，mixed两个裁剪同照片不独立。数据food_multiclass_probe_v1，脚本run_food_multiclass_probe.py，先读食材多类别与非食物压力验证报告.md；未训练不采用无后台。下一步局部与上下文矛盾保守复核，不能用整图菜覆盖局部真肉，先配对真肉真菜非食物再验证。

> 2026-10-27新视频+食材类别功能完成：review_board_video.py可选--food-categories复用4B，仅board_candidate裁剪(10%边距/边长>=20)输出肉菜混合其他未知，保存原文/坐标；默认关闭，不与replay混用，不接判罚。新BV1yz4y1d7iF全片范围固定10s采样12点：11有板，110s盛盘遮板漏；22候选裁剪19菜2肉1未知，逐图看2肉均50s小胡萝卜，不能报检测准确率。真实推理累计处理88s非实时；37测试+模式拒绝检查过，旧13回放12+1未知不变。先读新来源视频与食材类别联调报告.md。无训练/自动采用/后台，12点已消费；下一步新源肉菜/非食物误框、局部与上下文矛盾需复核，勿用整图菜覆盖局部肉或改提示追本片分数。

> 2026-10-27短视频入口已交付：review_board_video.py一次加载4B、固定区间间隔采样（最多120点）、严格校验，保存原帧/原文/JSON/CSV/中文报告/联系图；有partial/status原子进度。--replay绑定视频/每帧SHA，无VLM推理，错源/缺帧拒绝、错帧unknown。真实3点(78/80/82)跑通，旧13点回放12候选1未知精确复现food[2]错误；31单元测试+3CLI边界通过，报告/联系图回放通过，历史SHA不变。非实时/全帧/错用或新增准确率，无分类头删框/平滑/训练，无后台。先读短视频复核入口交付说明.md及短视频复核工具用法.md；下一步新整源短片可用性验证，再设计板上食材类别关联。

> 2026-10-27结构校验工程已完成：region_response_validation.py严格校验、错误精确路径、原文保留；review_board_regions.py接入unknown/no_board_detected/candidate_present，新--response-file离线回放，生成raw_response.txt/中文validation_report.md，有效才叠框。真实10秒错误定位$.food[2]且不展平补分；合法回放/可选头入口/实际4B推理通过。25测试过，审计已存56响应54有效2失败，新旧判定无差异、源SHA不变。非识别准确率提升，不训练/无后台。先读结构化响应校验改进报告.md。下一步复核/时间片异常展示与新整源验证，勿自动删框或平滑伪造成功。

# 给 dsh 的接手提示词

你接手用户的“用大模型识别后厨案板错用”项目。请先读本文件，再读同目录交接提示词.md的最新开头和连续片段采样诊断报告.md。用户要求自主推进、一次多做一些，不要反复要求他提供已有文件或确认常规操作。当前明确要求交接，不要再把工作转成GitHub上传。所有结果保守、据实，不能把开发样本说成现场准确率。

## 1. 工作位置和环境

主目录 D:/CodexProjects/board-misuse；脚本/数据/报告在outputs，原始数据/环境/权重在work，视频在待导入视频。C盘原工作区的outputs/work是D盘Junction。尽量所有文件留D盘。

- 检测/训练Python：D:/CodexProjects/board-misuse/work/board-env/Scripts/python.exe。
- VLM Python：D:/CodexProjects/board-misuse/work/vlm-env/Scripts/python.exe。该环境用board_runtime.pth复用board-env CUDA包，transformers4.57.6；不要乱升级原训练环境。
- RTX5070Ti Laptop 12GB，64GB内存。Qwen3-VL-4B BF16/SDPA、512缩略图已实际运行，显存接近卡容量。勿同时开多个GPU模型。
- PowerShell；读写文本显式UTF8；Python中文输出设PYTHONIOENCODING=utf-8。
- 设置YOLO_CONFIG_DIR=D:/CodexProjects/board-misuse/work/yolo-config、YOLO_OFFLINE=True、YOLO_AUTOINSTALL=False。
- 冻结YOLO.embed特征必须显式rect=False、imgsz640：默认rect=True会因批内图像形状改变补边，从而使同图分类翻转！修正后仍有GPU小数值误差，但测试类别一致。
- 现在没有后台训练或待完成推理。不要重装环境或重复已完成实验。需要长训练时隐藏启动并保存日志。

## 2. 项目实际水平

现有默认仍以active_models.json为权威（6个key）；本阶段没有修改默认、没有采用实验候选。旧YOLO食材/案板模型及规则仍保留。

现已接入本地Qwen3-VL-2B/4B做冻结推理，**没有任何视觉语言大模型微调**。最近实际训练仅为原始YOLO11s冻结特征上的线性案板/桌面分类头，不是YOLO主干训练。

还不能自动判断案板错用；不能判断生熟、物理接触、清洗或换板。实际用途制度没有提供。输出needs_review，空检测、空板、无板都不能等于合规。

## 3. 可运行功能

outputs/review_board_regions.py：本地4B自动给板面/容器/食物粗框，程序按容器优先几何规则产生位置候选，保存result.json和review.jpg。--output-dir必须为新目录；不覆盖已有结果。

PowerShell示例（换图片及新输出目录）：

& 'D:/CodexProjects/board-misuse/work/vlm-env/Scripts/python.exe' 'D:/CodexProjects/board-misuse/outputs/review_board_regions.py' 'D:/你的图片.jpg' --output-dir 'D:/CodexProjects/board-misuse/outputs/新复核01'

可选--board-head-review会先释放4B显存，再给原始板候选添加两个分类头的未校准意见；**不删原始框**、不修改位置结果、始终needs_review，默认不启用。board_candidate_review.py负责意见计算。

food_region_gate.py：容器覆盖食物框>=0.8优先容器候选，部分重叠>0.2记uncertain，否则板覆盖>=0.8才板上候选。全为矩形启发式，不能证明接触。check_food_regions.py可只处理已有框JSON。

已完成CLI实际烟雾推理，不要把重复烟雾样本计入测试。test_food_region_gate+test_misuse_rules共13项通过。用法见区域识别实验用法.md。

## 4. 模型和训练结果

4B权重：work/models/Qwen3-VL-4B-Instruct，约8.88GB；官方修订ebb281ec70b05090aa6165b016eac8ec08e71b17。下载与权重SHA在outputs/experiments/qwen4b_frozen_baseline。2B同名模型目录也存在。

正式修正后的分类头：
- experiments/board_table_rectfixed_v1/head_last.pt：18裁剪训练，旧8验证5/8。
- experiments/board_table_rectfixed_v2/head_last.pt：46裁剪训练，旧8验证6/8、新13验证12/13。旧案板召回1/3→3/3，但桌面正确4/5→3/5；新13小头和扩充头都是12/13，没提升。
- 小头视频板面3/4、木桌3/3；扩充头板面4/4、木桌2/3。有取舍，不采用。

board_table_crops_v2/manifest.json共67张（训练46、旧验证8、新留出13）。来自对象框裁剪，不能说整图无案板。官方LVIS验证586含桌面图片中，只有3张明确无案板且没有CCBY，不能用“没标案板”充当无板负例。按元数据CCBY使用、源图ID/SHA无跨组重复，摄影者/场景分组未证实，COCO预训练重叠未知。

原未固定rect的board_table_linear_pilot_v1、board_table_linear_expansion_v2只作历史，不应拿来部署。

## 5. 最近几批验证（均已消费开发）

阅读对应报告及locked_plan/原始results，不要仅引用成功批次：

- 新视频正负对照与复核功能报告.md：2新视频10图，4B有板3/4、无板误3/6；小头过滤有板3/4误1/6；扩充头无改善。
- 同厨房木台面与案板对照报告.md：2新厨房10图，2近邻案板点检出、8无板不误；两头都保留真板。但不是10次独立事件。
- 三批选定开发26帧汇总（排除预先不确定4帧）：4B TP5/FN1/FP5/TN15，见experiments/qwen4b_presence_development_summary.json。这不是自然场景准确率。

**最新连续片段采样诊断报告.md：**
- temporal_presence_v1：两个已有视频各24秒窗口，每2秒采样26张；不是全帧/实时推理，非新来源测试，部分点和旧开发图重合，不可加总成更多独立样本。
- counter_transition(70–94s)：6有板7无可见板，4B/两头都与采样标签一致。
- prep_to_stove(0–24s)：13有板，4B12正常+1schema未知；小头9正常+3漏板+1未知；扩充头12正常+1未知。
- 小头漏0/16/18秒，其中16、18是相邻2采样点，不能说证明连续漏4秒。
- 10秒原始JSON合法，但food一个框多套一层列表，导致严格schema失败；整条unknown。**未清洗补分、未沿用前帧、未做平滑**。
- 输出experiments/qwen4b_temporal_presence_v1：results.json、temporal_summary.json、timeline.csv、timeline.png、两窗口叠框接触表（均已看）；头对照experiments/temporal_head_filter。
- 脚本evaluate_qwen4b_temporal.py、evaluate_temporal_head_filter.py、summarize_temporal_presence.py。

## 6. 下一步优先工作

先处理结构化结果的可诊断性，而不是继续补同类单帧分数：
1. 为坐标schema校验增加明确字段/路径/错误类型，区分JSON不合法、坐标多层嵌套、越界、零面积、超数量。
2. 原始响应永久保留；识别失败明确unknown/needs_review，不能转成无板、空板或合规。若探索分字段隔离，只可作为新版本候选输出，必须保留完整失败状态，不能改写历史指标或默默展开嵌套列表。
3. 对真实10秒失败和合法/越界/嵌套/空列表等案例做必要测试，接入可选图片复核入口，完成实际运行验证与文档。
4. 保留小头仅意见的限制，禁止自动删框。暂不靠平滑延用上一帧掩盖漏检，尤其镜头剪辑前后不能直接认作同一案板。
5. 后续再做新整源视频、不同遮挡/木台面对照；先锁标签/配置，区分开发与新留出。不要针对已消费26点改提示词、阈值、轮数追分。

## 7. 其他交接约束

公开可访问视频不代表获训练/发布授权；目前仅本地研究，未上传媒体或权重。官方LVIS来源许可来自元数据，保留原Flickr链接，作者未独立核实。助手标注非专家，粗框非掩膜/真实接触GT。

GitHub https://github.com/duom1e/board-misuse 最近已上传版本为61c2505（2026-10-26），**后续VLM与分类头/时序新进展尚未同步**。当前用户要交接，不要自行变成上传任务；若以后明确要求上传，先检查远程新head再处理。项目根没有普通.git，之前通过GitHub连接器git-data API上传。原始图片/视频/权重不发布。

推进时每完成一组更新交接提示词.md和简明报告，中文说明实际改动、验证与限制。用户偏好自主完成、不停下来反复询问；缺现场材料时可做独立研究，但不要把研究诊断夸成完成现场交付。
