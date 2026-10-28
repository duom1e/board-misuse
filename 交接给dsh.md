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
