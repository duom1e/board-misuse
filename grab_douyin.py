"""抓取抖音视频到「待导入视频」，然后自动抽帧跑模型。

用法：
  python grab_douyin.py "https://v.douyin.com/xxxx/"        # 分享链接
  python grab_douyin.py "https://www.douyin.com/video/123"  # 视频页

前提：抖音需要"新鲜 cookie"。脚本会直接用你本机 Edge 里现有的抖音 cookie
（`--cookies-from-browser edge`）。Edge 正在运行时 cookie 数据库被锁，必须先关掉 Edge。

脚本只读本机 cookie 用于这一次下载，不上传任何东西；成功后自动调用 import_video.py。
"""
import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
INBOX = BASE.parent / '待导入视频'


def edge_running():
    try:
        output = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq msedge.exe'],
                                capture_output=True, text=True, timeout=20).stdout
        return 'msedge.exe' in output
    except Exception:  # noqa: BLE001
        return False


def inspect_cookies(path):
    """Netscape cookies.txt 里必须真的有抖音域名的条目，否则直接告诉用户哪里不对。"""
    try:
        text = Path(path).read_text(encoding='utf-8', errors='ignore')
    except Exception as exc:  # noqa: BLE001
        print(f'读不到 cookies 文件：{path}（{exc}）')
        return None
    names = []
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) < 7:
            continue
        domain, name = parts[0], parts[5]
        if 'douyin' in domain or 'bytedance' in domain or 'amemv' in domain or 'snssdk' in domain:
            names.append(name)
    print(f'cookies 文件里有 {len(names)} 条抖音相关记录：{", ".join(sorted(set(names))[:12]) or "无"}')
    if not names:
        print('这个文件里没有抖音 cookie。请确认：')
        print('  1) 导出格式是 Netscape cookies.txt（不是 JSON）')
        print('  2) 导出时浏览器正打开着 douyin.com 且已登录')
        print('  3) 扩展导出范围选了"当前站点"或"全部"')
        return None
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('--cookies', default=None,
                        help='从浏览器导出的 cookies.txt（推荐，绕过浏览器加密与锁）')
    parser.add_argument('--no-import', action='store_true', help='只下载，不自动抽帧')
    args = parser.parse_args()

    INBOX.mkdir(parents=True, exist_ok=True)
    if args.cookies:
        if inspect_cookies(args.cookies) is None:
            return 3
        cookie_args = ['--cookies', args.cookies]
    else:
        if edge_running():
            print('Edge 正在运行，cookie 数据库被锁，读不到抖音会话。')
            print('三个选择：')
            print('  A. 完全退出 Edge 后重跑（注意：新版 Edge/Chrome 的 cookie 用 app-bound 加密，可能仍解不开）')
            print('  B. 用扩展导出抖音 cookies.txt，然后：python grab_douyin.py "<链接>" --cookies 你的cookies.txt')
            print(f'  C. 抖音里「保存本地」，文件放进 {INBOX}，再双击 outputs\\导入视频.cmd')
            return 2
        cookie_args = ['--cookies-from-browser', 'edge']

    target = INBOX / '%(title).80s.%(ext)s'
    cmd = [sys.executable, '-m', 'yt_dlp', '--no-warnings', *cookie_args,
           '-f', 'mp4/best', '-o', str(target), '--print', 'after_move:filepath', args.url]
    print('下载中：', args.url, flush=True)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print('下载失败。常见原因：')
        print('  1) cookie 无效 / 未登录（先用浏览器打开一次该视频页）')
        print('  2) 链接不是 /video/<id> 或 v.douyin.com 形式（/shipin/<id> 不支持）')
        print('  3) 新版浏览器 cookie 加密解不开：改用 --cookies cookies.txt')
        return result.returncode
    if args.no_import:
        print('下载完成，未抽帧。')
        return 0
    print('下载完成，开始抽帧与检测 ...', flush=True)
    return subprocess.run([sys.executable, str(BASE / 'import_video.py')], text=True).returncode


if __name__ == '__main__':
    raise SystemExit(main())
