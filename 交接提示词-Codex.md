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

> 2026-10-30竹板新源真实训练完成：prepare_bamboo_chicken_boxes.py BV1PW411F7mD02/03两整图11框叠图全核，food_group_boxes_v4；train_food_bamboo_source_adaptation.py550图15轮固定last，同64val/held来源隔离核。food_bamboo_source_adaptation_v1原鸡6框匹配4(前黄皮1)，训练旧9框7，旧mAP50 .5677/.95 .4106，bg0/23 food8/19类6。鸡held改善但其他退步，保留不采用；原held已反复开发非独立测试。报告/last/source_integrity完整，无后台默认改变/VLM微调。下步保住4/6同时恢复其他food检出，不能只报4。

> 2026-10-29真实cls权重对照完成train_food_clsweight_detector.py548图15轮，args核cls.5→1其余关键设置/同训练图SHA/64val一致，controlled_settings存。food_clsweight_detector_v1训练9框9匹配，鸡held0/6，旧map50 .5872/.95 .4308，bg0/23 food10/19类8；比基线held1→0，拒绝采用。训练拟合不泛化，停止在这小批继续HSV/cls追分，下一步来源扩充检测框。report/last/results完整，无后台VLM微调默认切换。

> 2026-10-29用户要求实际训练，完成颜色增强YOLO全层15轮：train_food_color_augmentation.py548图与上一轮图/标签SHA多重集、64val一致，HSV .015/.4/.3 vs全0，fixedlast。food_color_augmentation_v1旧mAP50 .5788/.95 .4295，bg0/23 food13/19类10/19；鸡held正确框1/6→0/6训练9框6，拒绝采用。报告/controlled_inputs/last存，无后台默认改变VLM微调。用户不满频繁只诊断，后续优先成批训练但必须验收held。

> 2026-10-29整图/ROI/分块尺度诊断food_scale_diagnosis_v1：同黄皮YOLO2旧鸡原帧6GT，full几何2肉0/3候选，manual直接裁剪4肉1/5候选，固定800x600四角窗+同类NMS.5几何3肉0/6候选。6叠框联系图全看，分块多大框重复无肉改善，暂不接默认。manual此轮原帧直接裁剪，旧ROI JPEG重编码5/6，不能说模型退步或纯尺度因果。diagnose_full_frame_food_scale.py与报告.md，无训练默认切换后台。下一步固定编码/缩放评估，优先类别监督，非只加tile。

> 2026-10-29新review_detector_food_video.py显式weights3点90/115/140实跑smoke_v2，0/0/2候选，原类别保留无板/接触推断，最多20点JSON/HTML/SHA；smoke_v1画框污染后续crop bug已修独立副本并重跑，旧不作有效成绩。检测器类别复核视频入口说明.md，无训练默认切换后台，未浏览器验收。

> 2026-10-29完整混合检测+4B类别复核：evaluate_detector_vlm_category_chain.py6组ROI15GT+23bg，两YOLO实际58图推理及所有proposal原4B复核完成。原几何7/15、类2→复核7；yellow候选几何12、类8→复核12。同23bg原有4proposal=3unknown1仍food，yellow0proposal未触发bg复核，不能声称VLM负例全成功。原类别保留，假设替类统计不部署；detector_vlm_category_chain_v1报告/audit/results/plan，report_detector_vlm_chain.py。历史消费且部分训练图/同源，非独立准确率。无训练默认切换后台。下一步纠错链保留误报复核，不自动把unknown当正确过滤。

> 2026-10-29重要纠正鸡误差：chicken_error_decomposition_v1同2鸡ROI6GT，类别无关一对一IoU.5匹配5/6，肉类正确1/6；4几何框实际veg，1GT定位缺。不能再笼统说定位只有1。原4B对全部6真实proposal复核均meat，离线假设替类5/6但不部署(已知肉定向无负，手刀曾误肉)，6裁剪全看。diagnose_chicken_category_vs_location.py/报告.md，未训练无自动覆盖无后台。下一步类别混淆监督+非食物安全对照。

> 2026-10-29新下载BV1PW411F7mD251s/BV1Go4y1o7ym173s各12帧已看，后者黄皮清楚；food_group_boxes_v3两新ROI7框全看，train_food_yellow_chicken_adaptation.py548图15轮完成。旧mAP50 .5851/.95 .4363，背景0/23，food11/19类9/19；原鸡框仍1/6，bestmeatIoU 0/.008/.869/.120/0/0，拒绝采用。yellow源已训练消费不可盲测，另美食台源仅筛未训。报告/held_iou_diagnosis/raw权重留存，无后台默认未改无VLM微调。旧foodcheck2源24帧也筛但不适配排除。

> 2026-10-29另一鸡源补框真实训练完成：food_group_boxes_v2新增BV1hK411u73L01/04/08三ROI4框，06混边碗肉排除保留理由，非专家叠图全看。train_food_chicken_group_adaptation.py546图15轮，原鸡2ROI留出未训。训练旧9框7→8，鸡6框仍1，旧mAP50 .5883→.5722/.95 .4273，23bg0误，19food9检出8类，退步拒绝采用。报告food_chicken_group_adaptation_v1，权重/结果/计划存，无后台VLM微调默认未变。下一步黄皮小块/遮挡有针对跨源数据，不再白皮近景堆量。

> 2026-10-29新增4食材组ROI真实YOLO训练完成：train_food_group_adaptation.py 543(512旧16负11原正4组ROI)，15轮AdamW.0001固定last，同64val。原/512回放/新增组框：训练9框匹配2/4/7，鸡来源留出6框0/1/1，未匹配预测8/8/5；旧mAP50 .5777→.5883，.95 .4328→.4409；23背景0误，19food12检出9参考类。food_group_adaptation_v1含last/plan/groupresults/报告，report_food_group_adaptation.py SHA隔离核。训练拟合提升鸡源仍差，不默认采用；下步别的鸡肉来源检测框，勿把当前2鸡留出追分训练。无后台/VLM微调。

> 2026-10-29新检测框数据food_group_boxes_v1：6限定ROI15食材组框(猪5鸡6菜4)叠框全看，manifest非专家SHA来源，4猪/椒训练候选2鸡整源开发留出，非全帧实例/盲测。4检测模型固定.25/640rectFalse类别IoU.5一对一：原2/15 balanced1285/15 replay5125/15 positive3 3/15；未匹配预测8/3/8/6不全称FP。prepare_food_group_boxes.py/evaluate_food_group_boxes.py与报告.md，尚未用这6ROI训练；下步512回放+4ROI，保留鸡来源，勿继续曝同11。无后台训练默认未变。

> 2026-10-29真实YOLO第4轮正例曝光3完成：train_food_replay512_positive3.py 539独特/561条(512旧16负11正x3)，15轮固定last，同旧64val，源SHA与曝光次数核验。原/正1/正3：food检出11/11/9，参考类8/8/6，背景训/外源2→0→0，旧mAP50 .5777/.5865/.5855，.95 .4328/.4332/.4282。曝光增加失败，拒绝采用，停止调该批重复；下一步不同源检测框数据。food_replay512_positive3_detector_v1报告/comparison/last.pt，report_food_positive_exposure.py，默认未变，无VLM微调后台。

> 2026-10-29第三轮YOLO真实全层训练完成：train_food_replay512_detector.py旧128→512，同16负+11框正=539，15轮AdamW.0001固定last，同64val/源cropSHA断言无交。原/128平衡/512平衡背景FP2→0→0/16、外源2→0→0/7，19food检出11→11→11，参考类8→10→8；旧mAP50 .5777→.5414→.5865，.95 .4328→.4049→.4332。旧域恢复但类别改善回落，默认不变。food_replay512_detector_v1报告/comparison/last.pt，report_food_replay512.py。训练结束无后台，无VLM微调，历史开发不称盲测。

> 2026-10-29因用户要求转回真实检测训练：train_food_hard_negatives.py 128旧+16紧crop负144，15轮AdamW.0001全层固定last；train_food_balanced_detector.py同配置+11已有肉末检测框正155，两个真实训练完成。64相同旧val/trainSHA无交。原/负/平衡：训练背景FP2→0→0/16，训练外源背景FP2→0→0/7，19foodcrop有候选11→7→11，出现参考类8→6→10；旧mAP50 .5777→.5556→.5414。正负平衡开发改善但旧域退步，默认未改，无VLM微调后台。两个实验目录before/after/plan/last.pt完整，食材检测两轮训练对照.md与compare_food_detector_training.py。后续扩大检测框正例与旧域回放，不只加负。先前视频辣椒5点pipeline完成但未审计，不冒称效果。

> 2026-10-29成对意见进HTML/events：build_misuse_demo.crop_pair_html展示双类别/坐标/关系，events.crop_pair_reviews保留，旧输入无段。video_food_pairs_evidence_v2真实74s旧结果重导3组核，状态needsreview，3新展示测试，无训练推理后台，未浏览器验收。成对裁剪证据页说明.md。

> 2026-10-29成对裁剪接视频CLI与wrapper --food-crop-pairs默认关requiresfoodcategories，49s直接和74s完整pipeline实跑成功，27测试+依赖拒绝不建目录通过。视频成对裁剪复核说明.md；pair存JSON/report，HTML仍原类别不声称双展示，无训练采用后台。

> 2026-10-29分类helper可选compare_tight_crop默认关，实际4可疑候选8推理，帧0手刀肉肉仍错；不能把一致当正确。27测试通过，无训练采用后台，报告food_candidate_pairs_v1/报告.md。仅Python参数未接视频CLI。

> 2026-10-29裁剪敏感性完成：food_crop_context_v1 12对24实推，3页全看，4变化(3手扩边引入食材unknown转肉菜，蒜扩边菜转unknown)，不是4误分类；原prompt/4B不变，实验margin50%非默认10%。报告.md/audit.json。后续标注实际检测宽框与紧ROI分开审核，背景标签不能扩边继承，无训练采用后台。

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

> 2026-10-27时间片诊断完成：两段各24s、2s采样26点（已有视频、部分旧点，非全帧/独立事件/实时验证）。19有板7无板：4B与扩充头18正常+1schema未知，无板误0；小头额外漏0/16/18秒3点，最长2个相邻采样漏板，不能说连续漏4秒。10秒JSON food坐标多层嵌套被严格拦截，未清洗补分/延用前帧。时间轴与两叠框表已看。先读连续片段采样诊断报告.md；禁止自动用小头删框，现可选功能仍仅展示意见。未训练不采用无后台。下一步结构化错误定位/失败展示（unknown与无板分开），不做平滑掩盖错误。

> 2026-10-27同厨房对照完成：2新木厨房视频，先看26候选锁定10（同厨房2近邻阳性+4台面阴性，另一厨房4阴性）。固定4B板2/2，无板误0/8，两头保留2候选，全部输出/叠框复核通过，非10/10错用或独立事件准确率。跨3批选定开发26帧，原4B TP5/FN1/FP5/TN15；4白面不确定帧预先排除。先读同厨房木台面与案板对照报告.md；数据same_kitchen_wood_v1，原始experiments/qwen4b_same_kitchen_v1。未训练/不自动采用无后台，10图已消费；下一步冻结参数做连续短片/时间片检验，别只挑帧报成功。

> 2026-10-27新视频对照+功能完成：2新视频24帧先复核锁定10（同做饭视频4板+2餐桌无板，木工4无板）。固定4B检板3/4、无板误3/6；小头过滤3/4且误1/6，扩充头无改善3/6。只图像存在性非AP/错用准确率。新可选review_board_regions.py --board-head-review已真实跑通，附两头未校准意见，原框全保留needs_review，默认关闭；13测试过。不训练不自动采用无后台。先读新视频正负对照与复核功能报告.md，10图已消费；下一步同厨房木质台面/可移动案板区域对照，防背景捷径。

> 2026-10-27扩充+修正完成：新增候选62复核留41，总67=46训+旧8+新13。发现YOLO embed默认rect随batch shape变化：同木桌概率.2365→.6149翻转；固定rect=False重训小/扩充两个头，类别跨批一致仍有GPU小误差。正式实验board_table_rectfixed_v1/v2：旧8 5→6，板1/3→3/3桌4/5→3/5；新13两头均12/13(板6/7桌6/6)。视频粗板4处旧3新4，木桌旧3新2，退步不采用，无后台/无VLM微调。先读案板桌面扩充与预处理修正报告.md；以后嵌入必须rect=False，原默认rect实验只历史；67/新13现已消费，不改轮数阈值追分。下一步配对覆盖/视角难例+新视频整源验证，默认不变。

> 2026-10-27实际线性头训练完成：先核LVIS桌面586图仅3明确无板且无CCBY，不可将未标板当负。38对象裁剪复核保留26(10板16桌)，源照片隔离train18/val8。原始yolo11s冻结特征+线性头固定120轮，训练18/18、验证5/8（板1/3，桌4/5），已暴露3木桌全图3判桌但不能抵消板漏检。候选不采用，不训练YOLO主干/VLM，无后台。先读案板桌面分类头训练报告.md，权重experiments/board_table_linear_pilot_v1/head_last.pt，数据board_table_crops_v1；不得用此头否决4B案板，别改阈值/分组追8图分数，下一步配对食物覆盖/材质/视角数据与场景隔离。

> 2026-10-27新来源区域验证完成：新增3视频，36帧先复核锁定10图，固定4B定位参数只跑一次。6无板负例中2张木桌误板（木桌3张中2）；木桌食物0误报；沙拉容器错位/漏检导致uncertain；4白色操作面不确定压力图不评分，其中1输出截断解析失败。9/10可解析，不采用无后台。先读4B区域流程新来源负例报告.md，数据region_new_sources_v1，结果experiments/qwen4b_new_sources_v1；10图现已消费，别调参追分或报部署准确率。下一步配对木桌/案板+容器区域标注与整源验证，暂无用户操作。

> 2026-10-27大步推进完成：4张粗区域/8食物框逐图复核；保守geometry gate(容器优先)+CLI工具完成，13软件测试通过。4B(修订ebb281...)本地成功，固定对照支持集合2→7/10，旧双字段3→10/11，均消费开发非独立准确率。4B自动区域4图已跑并看叠框：盘肉container候选、板上肉菜board候选，但盆混手/盘下缘漏框，不能报4/4成功。review_board_regions.py实际CLI推理通过，needs_review无合规判定。未训练不采用无后台。先读区域关联原型与4B对照报告.md及区域识别实验用法.md；下一步固定路线新来源/木桌/板外/遮挡独立区域验证。

> 2026-10-27：food_support_relation_v1完成10图图像级位置标签/接触表复核（非专家），盘盆遮板时board_food改uncertain，不回写旧成绩。Qwen新直接承托任务固定一次：严格集合2/10、格式9/10，常混淆手/桌面托容器与直接托食物，板上菜误容器。training_ready=false、未微调不采用无后台。先读食材承托位置标注与基线报告.md；下一步多区域标注+独立来源/木桌负例，不再追同批提示词分数。

> 2026-10-27：容器关联反例11图完成，固定上一轮模型/提示词。原始board11一致但7无板schema失败；有板辣椒盆none正确，装盘鸡肉误报板上mixed；2已消费正对照正确。严格有效4/11、严格双字段一致3/11，不采用无后台。先读碗内食材与案板关联反例报告.md；下一步位置关联标注/约束，木桌负例尚未补，别重复食材类别混训或改提示词追分。

> 2026-10-27：案板存在性新任务开发诊断完成，同已消费15图原始board字段15一致，但2无板图food_on_board违反not_applicable约定，严格输出仅13/15，已拦截为invalid/needs_review。不是独立15/15现场成绩，不采用无后台。下一步补板外/碗内食物与无板木桌对照，先读视觉大模型案板关联诊断报告.md；脚本evaluate_qwen_board_relation.py。

> 2026-10-27：Qwen新来源固定提示词验证完成，15图原14一致、遮底部15%后13一致；鸡肉10/10稳定、空板3/3→2/3，无板水盆2图中1个误empty。不能用empty认定案板存在/合规。未训练、不采用、无后台。下一步验证案板存在性/区域关联，先读视觉大模型新来源验证报告.md；原始结果experiments/qwen_new_sources_v1。

> 2026-10-27：首次本地Qwen3-VL-2B基线已完成（非训练），36已消费开发图类别33一致；11训练图11/11、已释放红椒5图4/5、B18菜18/18、2空均unknown。整图类别不是定位TP或独立准确率。不采用、无后台。下一步固定模型/提示词补新来源负例与肉类验证。先读本地视觉大模型食材基线报告.md，脚本evaluate_qwen_food.py，环境work/vlm-env。

> 2026-10-27最新：minced_replay12_trial已完成，曝光3→12，新11训练拟合11/11，HF15 TP8FP2FN3过操作门槛，旧64mAP50 .5740过回归。保留来源按计划首次释放，5图6菜区域两个候选均0检出，跨来源失败，不采用无后台。这5图现已消费，不能再当新鲜测试。先读细碎样本混训曝光对照报告.md；暂停同类循环混训，下一步比较更强视觉识别基线。

> 2026-10-27最新诊断完成：minced_training_fit_check只用新11图做记忆/拟合校验（train=val），120轮固定last，原TP2→TP11/FP0/FN0。遮底部15%且不碰GT后同11图仍全匹配。不是独立准确率，不采用；没有评保留来源/B20/旧64，默认不变、无后台。先读细碎形态小样本拟合校验报告.md。下一步查混训采样/优化步进和选型，别继续盲补数据或把11/11说成现场成功。

> 2026-10-27最新训练完成：minced_shape_adaptation_v1_reviewed，591独特/645条目，30轮freeze0，选epoch20.pt。旧64mAP50 .5636过回归，但操作门槛失败；旧B20 TP1FP14FN17、13肉预测，未修好。新11图拟合仅4/11（last事后6/11），缓存确认全部标签正确加载3次。保留来源5图6菜框未释放、没跑模型，候选不采用无后台。先读细碎形态补样混训实验报告.md。

> 2026-10-27最新数据：已补细碎形态对照，minced_shape_annotation_v1共17图18框，逐图复核。11训练候选来自BV1Ys411L7r6同一案板（5肉6菜）；另BV14K4y1x7LF的6菜图整源保留验证候选，未跑模型，其中44秒高遮挡暂不入严格IoU。没有新训练；先读细碎红椒与肉末对照素材说明.md。

> 2026-10-27最新：解除冻结受控比较已完成（redveg_contrast_unfrozen_control，无后台）。其他设置/输入同freeze10；新16训练图匹配16→21/26，双机位TP7FP1FN4过操作门槛，旧64图mAP50 .5826过回归门槛；B站18菜区域仍TP1FP19FN17、16肉预测，混淆未解，候选不采用。先读骨干冻结策略对照实验报告.md，不再重复调整冻结参数。

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
