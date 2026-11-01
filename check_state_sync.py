"""工作对账：扫描 outputs 里近 N 小时的新产物，列出《项目状态总表.md》里没有记录的条目。

只把"关键产物"（目录 + 报告 .md）列为待补记；脚本/日志/中间 json 只统计数量，避免噪音。

用法：
  python check_state_sync.py                 # 默认近 24 小时
  python check_state_sync.py --hours 72
  python check_state_sync.py --fail-on-missing   # 有未记录的关键产物时返回码 1
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent
TABLE = BASE / '项目状态总表.md'
# 这些不是"产物"，是工具与日志
AUX_SUFFIXES = {'.py', '.log', '.err.log', '.cmd', '.txt', '.csv'}


def classify(path):
    if path.is_dir():
        return 'artifact'
    if path.suffix.lower() == '.md':
        return 'artifact'
    if path.suffix.lower() in AUX_SUFFIXES:
        return 'aux'
    return 'aux'


def recent_entries(hours):
    cutoff = datetime.now() - timedelta(hours=hours)
    entries = []
    for path in BASE.iterdir():
        if path.name.startswith('.') or path.name == '__pycache__':
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        if modified < cutoff:
            continue
        entries.append({'name': path.name, 'kind': classify(path), 'when': modified,
                        'is_dir': path.is_dir()})
    return sorted(entries, key=lambda item: item['when'], reverse=True)


def mentioned(name, text):
    return name in text or name.rsplit('.', 1)[0] in text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=24)
    parser.add_argument('--fail-on-missing', action='store_true')
    args = parser.parse_args()

    if not TABLE.exists():
        print(f'缺少总表：{TABLE}')
        return 2
    text = TABLE.read_text(encoding='utf-8')

    entries = recent_entries(args.hours)
    artifacts = [e for e in entries if e['kind'] == 'artifact']
    aux = [e for e in entries if e['kind'] == 'aux']
    missing = [e for e in artifacts if not mentioned(e['name'], text)]

    print(f'总表：{TABLE.name}（{len(text.splitlines())} 行）')
    print(f'近 {args.hours:g} 小时：关键产物 {len(artifacts)} 个，工具/日志 {len(aux)} 个')
    print(f'关键产物中未被总表记录：{len(missing)} 个')

    if missing:
        print('\n待补记（建议进 B/C 栏或对应轮次）：')
        for entry in missing:
            tag = '[目录]' if entry['is_dir'] else '[报告]'
            print(f"  {entry['when']:%m-%d %H:%M}  {tag}  {entry['name']}")
    else:
        print('\n全部关键产物已在总表中。')

    covered_aux = sum(1 for e in aux if mentioned(e['name'], text))
    print(f'\n工具/日志覆盖率：{covered_aux}/{len(aux)}（辅助文件不强制逐条登记）')

    if missing and args.fail_on_missing:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
