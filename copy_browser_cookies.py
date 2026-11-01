"""Copy the browser's douyin cookies into a fresh SQLite DB that yt-dlp can read.

The live browser keeps its Cookies file locked, so the original is opened read-only with
sqlite's immutable flag and the douyin rows are written into a new database laid out like an
Edge profile. Only douyin-related rows are copied; nothing is uploaded.
"""
import shutil
import sqlite3
from pathlib import Path

SRC = Path(r'C:\Users\13442\AppData\Local\Microsoft\Edge\User Data\Default\Network\Cookies')
LOCAL_STATE = Path(r'C:\Users\13442\AppData\Local\Microsoft\Edge\User Data\Local State')
PROFILE = Path.home() / 'AppData' / 'Local' / 'Temp' / 'dy' / 'edgeprof' / 'User Data'
DEST = PROFILE / 'Default' / 'Network' / 'Cookies'


def main():
    (PROFILE / 'Default' / 'Network').mkdir(parents=True, exist_ok=True)
    shutil.copy2(LOCAL_STATE, PROFILE / 'Local State')
    if DEST.exists():
        DEST.unlink()
    from urllib.parse import quote
    uri = 'file:' + quote(SRC.as_posix()) + '?immutable=1'
    source = sqlite3.connect(uri, uri=True)
    schema = source.execute(
        "select sql from sqlite_master where type='table' and name='cookies'").fetchone()[0]
    rows = source.execute(
        "select host_key,name,value,encrypted_value,path,expires_utc,is_secure,is_httponly,"
        "samesite,priority,has_expires,is_persistent,source_scheme,source_port,last_access_utc,"
        "creation_utc,last_update_utc,has_cross_site_ancestor,source_type "
        "from cookies where host_key like '%douyin%' or host_key like '%bytedance%' "
        "or host_key like '%amemv%' or host_key like '%snssdk%'").fetchall()
    source.close()
    dest = sqlite3.connect(DEST)
    dest.execute(schema)
    dest.executemany(
        'insert into cookies (host_key,name,value,encrypted_value,path,expires_utc,is_secure,'
        'is_httponly,samesite,priority,has_expires,is_persistent,source_scheme,source_port,'
        'last_access_utc,creation_utc,last_update_utc,has_cross_site_ancestor,source_type) '
        'values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    dest.commit()
    dest.close()
    print('rows', len(rows))
    for row in rows:
        print(' ', row[0], row[1], 'encrypted' if row[3] else 'plain')


if __name__ == '__main__':
    main()
