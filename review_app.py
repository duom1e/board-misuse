"""Local-only image review application. Run with the project's Python environment."""
import os
os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
os.environ.setdefault('YOLO_OFFLINE', 'True')
os.environ.setdefault('YOLO_CONFIG_DIR', r'D:\CodexProjects\board-misuse\work\yolo-config')
import json
import queue
import threading
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk
from annotation_io import prepare_image, normalized_box, export_annotation

BASE = Path(__file__).resolve().parent
RULES = {'肉类专用': ['meat'], '蔬菜专用': ['vegetable'], '允许肉类和蔬菜': ['meat', 'vegetable']}


class ReviewApp:
    def __init__(self, root):
        self.root = root
        root.title('案板图片复核 · 本地试验工具')
        root.geometry('1180x850')
        self.folder = None
        self.boxes = []
        self.models = {}
        self.events = queue.Queue()
        self.busy = False
        self.rule = tk.StringVar()
        self.model = tk.StringVar(value='当前默认')
        self.group = tk.StringVar()
        self.decision = tk.StringVar()
        self.status = tk.StringVar(value='打开图片后，在图上拖动框选案板。所有文件保存在D盘。')
        bar = ttk.Frame(root); bar.pack(fill='x', padx=10, pady=6)
        self.buttons = []
        for text, command in [('打开图片', self.open_image), ('撤销最后一个框', self.undo), ('清空框', self.clear), ('检查人工样本池', self.audit_pool)]:
            button = ttk.Button(bar, text=text, command=command); button.pack(side='left', padx=4); self.buttons.append(button)
        ttk.Label(root, text='试验提示：用途需要由现场规定确认；系统仅提示疑似冲突或待复核，不能判断生熟及真实接触。').pack(anchor='w', padx=14)
        self.canvas = tk.Canvas(root, width=1100, height=560, background='#222222', highlightthickness=0)
        self.canvas.pack(padx=10, pady=6)
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.drag_start = None
        options = ttk.Frame(root); options.pack(fill='x', padx=12)
        ttk.Label(options, text='这张图的案板用途：').pack(side='left')
        ttk.Combobox(options, textvariable=self.rule, values=list(RULES), state='readonly', width=20).pack(side='left')
        ttk.Combobox(options, textvariable=self.model, values=['当前默认', 'LVIS实验候选'], state='readonly', width=18).pack(side='left', padx=6)
        for text, command in [('按手框区域识别', lambda: self.inspect(True)), ('自动找案板并识别', lambda: self.inspect(False))]:
            button = ttk.Button(options, text=text, command=command); button.pack(side='left', padx=4); self.buttons.append(button)
        row = ttk.Frame(root); row.pack(fill='x', padx=12, pady=8)
        ttk.Label(row, text='来源组（同视频/连拍填同名）：').pack(side='left')
        ttk.Entry(row, textvariable=self.group, width=24).pack(side='left')
        ttk.Combobox(row, textvariable=self.decision, values=['已检查整图，所有案板均已框出', '已检查整图，确认没有案板'], state='readonly', width=34).pack(side='left', padx=8)
        button = ttk.Button(row, text='导出人工案板标注', command=self.export); button.pack(side='left'); self.buttons.append(button)
        ttk.Label(root, textvariable=self.status, wraplength=1120).pack(anchor='w', padx=14, pady=5)
        ttk.Label(root, text='手框方式：每块案板拖一个框；本版一张图的用途规则统一。标注只进入待整理池，不会自动加入训练。').pack(anchor='w', padx=14)
        root.after(100, self.poll)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[('图片', '*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff'), ('所有文件', '*.*')])
        if not path: return
        try:
            self.folder, self.im, self.metadata = prepare_image(path, BASE/'review_results')
            self.boxes = []; self.decision.set(''); self.group.set(''); self.rule.set('')
            self.scale = min(1100/self.im.width, 560/self.im.height, 1)
            self.display_size = (max(1, round(self.im.width*self.scale)), max(1, round(self.im.height*self.scale)))
            self.photo = ImageTk.PhotoImage(self.im.resize(self.display_size))
            self.redraw()
            self.status.set(f'已打开 {Path(path).name}；请框选案板，选择用途。')
        except Exception as exc: messagebox.showerror('打开失败', str(exc))

    def redraw(self):
        self.canvas.delete('all')
        if not self.folder: return
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        w, h = self.display_size
        for i, (x1,y1,x2,y2) in enumerate(self.boxes, 1):
            self.canvas.create_rectangle(x1*w,y1*h,x2*w,y2*h, outline='#00ffff', width=2)
            self.canvas.create_text(x1*w+5,y1*h+10,text=str(i),fill='#00ffff',anchor='w')

    def press(self, event):
        if self.folder and not self.busy:
            self.drag_start = (event.x, event.y)

    def drag(self, event):
        if self.drag_start:
            self.canvas.delete('drag')
            self.canvas.create_rectangle(*self.drag_start,event.x,event.y,outline='yellow',width=2,tags='drag')

    def release(self, event):
        if not self.drag_start: return
        try:
            self.boxes.append(normalized_box(self.drag_start, (event.x,event.y), *self.display_size))
            self.decision.set('')
        except ValueError as exc: self.status.set(str(exc))
        self.drag_start = None
        self.redraw()

    def undo(self):
        if self.boxes: self.boxes.pop()
        self.decision.set(''); self.redraw()

    def clear(self):
        self.boxes.clear(); self.decision.set(''); self.redraw()

    def inspect(self, manual):
        if self.busy: return
        if not self.folder or self.rule.get() not in RULES:
            messagebox.showinfo('还差一步', '请打开图片并明确选择案板用途。'); return
        if manual and not self.boxes:
            messagebox.showinfo('还差一步', '请先框出案板。'); return
        config = {'demo_not_site_policy': True, 'food_model': 'foodseg',
                  'board_model': 'board_lvis' if self.model.get() == 'LVIS实验候选' else 'board'}
        allowed = RULES[self.rule.get()]
        if manual:
            config['boards'] = [{'id': f'board_{i+1}', 'xyxy_normalized': list(box), 'allowed_foods': allowed} for i,box in enumerate(self.boxes)]
        else: config['all_detected_boards_allowed_foods'] = allowed
        source = self.folder/'image.png'
        destination = self.folder/('inspection_'+uuid.uuid4().hex)
        self.busy = True
        for button in self.buttons: button.config(state='disabled')
        self.status.set('正在本机识别，首次加载模型需要稍等……')
        def work():
            try:
                from check_board_misuse import inspect_image
                report = inspect_image(source, config, destination, self.models)
                self.events.put(('ok', report, destination))
            except Exception as exc: self.events.put(('error', str(exc), destination))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            kind, result, folder = self.events.get_nowait()
            self.busy = False
            for button in self.buttons: button.config(state='normal')
            if kind == 'error':
                self.status.set('识别失败：'+result)
            elif kind == 'audit':
                self.status.set(f'样本池：有效图片 {len(result["records"])} 张，来源组 {len(result["groups"])} 个，错误 {len(result["errors"])} 项。报告：{folder}')
                os.startfile(str(folder))
            else:
                status = '疑似用途冲突' if result['overall_status']=='suspected_misuse' else '待人工复核'
                self.status.set(f'{status}；案板区域 {len(result["board_candidates"])} 个。证据图与记录：{folder}')
                os.startfile(str(folder))
        except queue.Empty: pass
        self.root.after(150, self.poll)

    def audit_pool(self):
        if self.busy: return
        self.busy = True
        for button in self.buttons: button.config(state='disabled')
        self.status.set('正在检查人工样本池……')
        destination = BASE/'annotation_audits'/uuid.uuid4().hex
        def work():
            try:
                from prepare_manual_dataset import pool_audit, dump
                report = pool_audit(BASE/'annotation_pool')
                destination.mkdir(parents=True, exist_ok=False)
                dump(destination/'pool_audit.json', report)
                dump(destination/'group_assignment.template.json', {
                    'pool_fingerprint': report['fingerprint'],
                    'groups': {g: None for g in report['groups']},
                    'reviewer': '', 'review_notes': ''})
                (destination/'说明.txt').write_text(
                    f'有效图片 {len(report["records"])} 张，来源组 {len(report["groups"])} 个，错误 {len(report["errors"])} 项。\n'
                    '没有自动拆分或启动训练。重复候选、无效标签和来源组需在训练前审核。\n', encoding='utf-8')
                self.events.put(('audit', report, destination))
            except Exception as exc: self.events.put(('error', str(exc), destination))
        threading.Thread(target=work, daemon=True).start()

    def export(self):
        if not self.folder: return
        decision = {'已检查整图，所有案板均已框出': 'all_boards_boxed', '已检查整图，确认没有案板': 'confirmed_no_board'}.get(self.decision.get())
        try:
            folder = export_annotation(self.folder/'image.png', self.metadata, self.boxes,
                                       self.group.get(), decision, BASE/'annotation_pool')
            self.status.set(f'人工标注已保存，尚未加入训练：{folder}')
        except Exception as exc: messagebox.showerror('尚不能导出', str(exc))


if __name__ == '__main__':
    root = tk.Tk()
    ReviewApp(root)
    root.mainloop()
