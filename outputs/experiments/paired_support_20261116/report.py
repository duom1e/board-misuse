import json
from PIL import Image, ImageDraw, ImageFont

from evaluate import ARMS, inputs
from train import HERE, read

NAMES={"original":"原候选","food_11":"对照 11","paired_11":"配对 11","food_42":"对照 42","paired_42":"配对 42"}


def make_picture(models,vlm):
    import cv2
    wanted=[("kitchens","BV1Mg411U7SV_098.6s.jpg"),("transfer","2215"),("kitchens","BV1Zq4y1j7mm_121.6s.jpg")]
    images={(r["group"],r["name"]): (r,Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
            for r,image in inputs() if (r["group"],r["name"]) in wanted}
    font=lambda size: ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",size)
    sheet=Image.new("RGB",(1060,840),"#f9f8f5")
    draw=ImageDraw.Draw(sheet)
    draw.text((20,12),"同一画面，三种案板定位结果",font=font(28),fill="#17222d")
    draw.text((20,55),"绿色为人工参考边界，红色为模型候选。检测器门槛 0.25，Qwen 使用原有提示词。",font=font(17),fill="#334455")
    for col,(arm,title) in enumerate([("paired_11","配对训练 · 种子 11"),("paired_42","配对训练 · 种子 42"),("vlm","Qwen3-VL-4B")]):
        draw.text((20+col*350,90),title,font=font(20),fill="#17222d")
        rows=vlm["rows"] if arm=="vlm" else models[arm]["rows"]
        for idx,key in enumerate(wanted):
            label,original=images[key]
            im=original.copy()
            row=next(r for r in rows if (r["group"],r["name"])==key)
            boxes=row["boxes"] if arm=="vlm" else [b["xyxy"] for b in row["boxes"] if b["score"]>=.25]
            ctx=ImageDraw.Draw(im)
            if label["gt"]:
                ctx.rectangle(label["gt"],outline="#1ae595",width=max(4,im.width//100))
            for box in boxes:
                ctx.rectangle(box,outline="#f75045",width=max(3,im.width//180))
            im.thumbnail((338,188))
            x,y=20+col*350,130+idx*231
            sheet.paste(im,(x,y))
            subject=("酸菜鱼，无可见板","切青菜，圆形案板","酱汁餐盘，无可见板")[idx]
            draw.text((x,y+192),subject+f" · {len(boxes)} 框",font=font(16),fill="#17222d")
    sheet.save(HERE/"案例对照.jpg",quality=92)


def main():
    models={arm:read(HERE/"evaluation"/f"{arm}.json") for arm in ARMS}
    vid=read(HERE/"video_results.json")
    vlm=read(HERE/"vlm.json")
    regression=read(HERE/"vlm_regression.json")
    assert vlm["revision"]==regression["revision"] and vlm["prompt_sha256"]==regression["prompt_sha256"]
    all_vlm={g:v for run in (vlm,regression) for g,v in run["summary"].items()}
    lines=["# 板上食材与容器的配对训练","",
           "2026-11-17。复用上轮五张碗、盆和托盘负例，替换五张重复正例，用原视频中的不同画面标注案板边界。新正例来自三个视频，其中四张与已有容器负例来自相同视频。两组均有 285 条训练输入，250 条有板、35 条无板。每个种子只有五个输入位置及对应标签不同，其他 280 个输入和全部验证样本一致。",
           "","本次补齐三组各 15 轮训练，种子 11 的负例对照使用上轮已核验的第 15 轮权重。两组均从同一 YOLO11n 预训练权重开始，输入 640，未按评估结果挑选轮次。下表的原候选为 board_neg15_v1。",
           "","## 修订两厨房标签","",
           "逐张复看 20 张原图。酒店 42.2 秒左下角的圆砧墩由不确定改为有板。餐厅 121.6 秒是金属餐盘里的酱汁和面片，原记录将它当成了切番茄。另两张远景的物体用途难辨，改为不确定。修订后 4 张有板、14 张无可见板、2 张排除。3 张有板图能给出主案板参考框，剩下 1 张仅评存在性。原标签和本轮逐图改动分别保存在原项目和 geometry 目录。",
           "","## 检测器复测","",
           "位置命中要求有候选框与主案板参考框的 IoU 至少 0.5。仅存在性计数时，有候选即算检出。三段视频的配对检查使用 3 张有板图、4 张无板图；同源正例定位和全部同源负例静默同时满足，才算通过该来源。用于开发检查，部分来源以前曾用于其他任务。",
           "","| 模型 | 门槛 | 两厨房存在 /4 | 两厨房定位 /3 | 两厨房误报 /14 | 三视频定位 /3 | 三视频误报 /4 | 通过来源 /3 | 固定集定位 /14 | 固定集误报 /70 | 白圆板 /12 | 切鸡 /12 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in ("0.25","0.5"):
        for arm,model in models.items():
            m=model["summary"][t]
            k,v,d=m["kitchens"],m["transfer"],m["dev"]
            lines.append(f"| {NAMES[arm]} | {t} | {k['presence_hits']} | {k['located']} | {k['false_frames']} | {v['located']} | {v['false_frames']} | {sum(v['paired_source_pass'].values())} | {d['located']} | {d['false_frames']} | {m['white']['located']} | {m['chicken']['located']} |")
    lines.extend(["","两组检测器在五个新正例上均由对应对照的 3/5 定位到 5/5，门槛 0.25。固定开发集 70 张无板图上的误报在两个种子下均增加。新模型保留作实验。", "",
                  "## 食堂连续视频","","取同一段 40 秒、1200 帧视频。连续 5 帧中当前框均与前面各帧至少一个框重合 IoU ≥0.5 时提示。表内同时列出有框帧数，同一处设施可产生多个片段。","",
                  "| 模型 | 单帧候选段数 | 连续 5 帧候选段数 | 连续规则有框帧数 |","|---|---:|---:|---:|"])
    for arm,r in vid["arms"].items():
        v=r["windows"]
        assert r["weight_sha256"]==models[arm]["sha256"]
        lines.append(f"| {NAMES[arm]} | {v['1']['episode_count']} | {v['5']['episode_count']} | {v['5']['flagged_frames']} |")
    lines.extend(["","## 冻结视觉大模型","","本机 Qwen3-VL-4B 沿用项目原提示词、长边最多 512 像素的缩略图和贪心生成，先评 30 帧，再按相同设置扩到 84 张固定开发图和 24 个圆板锚点。两张不确定图保留原始回答但排除计分。原始框、归一化坐标换算及模型版本见 vlm.json 和 vlm_regression.json。","",
                  "| 画面 | 可解析 / 总数 | 有板检出 | 主案板定位 | 无板误报 |","|---|---:|---:|---:|---:|"])
    for group,title in (("kitchens","两厨房"),("transfer","三视频配对"),("dishes","容器特写"),("dev","固定开发集"),("white","白圆板"),("chicken","切鸡圆板")):
        s=all_vlm[group]
        d=models["food_11"]["summary"]["0.25"][group]
        ratio=lambda n,total: f"{n}/{total}" if total else "—"
        lines.append(f"| {title} | {s['valid']}/{s['total']} | {ratio(s['presence_hits'],d['positives'])} | {ratio(s['localized'],d['boxed'])} | {ratio(s['false_frames'],d['negatives'])} |")
    lines.extend(["","## 只将检测器候选送大模型复核","","用 84 张固定开发图做离线回放。YOLO 置信度门槛为 0.25，有候选才调用 Qwen。大模型也给出有效案板框才保留该帧的检测器框；位置仍按 YOLO 与主板参考框的 IoU ≥0.5 计分。两个模型判断的物体可能不同，此规则只用于检验图像级复核。","",
                  "| 检测器 | 需大模型复核 /84 | 保留正确位置 /14 | 无板仍提示 /70 | 送审后无法解析 |","|---|---:|---:|---:|---:|"])
    qrows={r["name"]:r for r in regression["rows"] if r["group"]=="dev"}
    cascade={}
    for arm,model in models.items():
        counts={"calls":0,"localized":0,"false_frames":0,"invalid":0}
        for r in model["rows"]:
            if r["group"]!="dev":
                continue
            candidates=[b for b in r["boxes"] if b["score"]>=.25]
            if not candidates:
                continue
            counts["calls"]+=1
            q=qrows[r["name"]]
            if not q["valid"]:
                counts["invalid"]+=1
            elif q["boxes"]:
                if r["gt"]:
                    counts["localized"]+=any(b["iou"]>=.5 for b in candidates)
                else:
                    counts["false_frames"]+=1
        cascade[arm]=counts
        lines.append(f"| {NAMES[arm]} | {counts['calls']} | {counts['localized']} | {counts['false_frames']} | {counts['invalid']} |")
    missed=[r["name"] for r in regression["rows"] if r["group"]=="dev" and r["gt"] and r["best_iou"]<.5]
    lines.extend(["", "固定开发集上全帧直接运行 Qwen 漏掉的主案板来自 "+ "、".join(missed)+"。这些视频和部分评估设置已在前期研究中使用过；同视频相邻帧也有关联，结果不能代表新厨房准确率。",
                  "", "下一步用不同厨房的连续片段验证全帧复核，记录漏掉的案板操作。案板错用还需要知道厨房给这块板指定的用途，并有食材接触证据，本次未评分违规事件。"])
    (HERE/"报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    compact={arm:{"model_sha256":r["sha256"],"metrics":r["summary"]} for arm,r in models.items()}
    (HERE/"summary.json").write_text(json.dumps({"detectors":compact,"vlm":all_vlm,"cascade":cascade},indent=2),encoding="utf-8")
    make_picture(models,vlm)


if __name__=="__main__":
    main()
