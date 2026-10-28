"""Deterministic frame schedule; never silently shorten an invalid request."""
import math
def plan_frames(start,end,interval,fps,frame_count,max_samples=120):
    if not all(math.isfinite(v) for v in [start,end,interval,fps]):raise ValueError('Sampling parameters must be finite')
    if start<0 or end<start or interval<=0 or fps<=0 or frame_count<=0 or max_samples<=0:raise ValueError('Invalid sampling range, interval or video metadata')
    count=math.floor((end-start)/interval+1e-9)+1
    if count>max_samples:raise ValueError(f'Request has {count} samples; limit is {max_samples}')
    rows=[{'requested_seconds':start+i*interval,'frame_index':round((start+i*interval)*fps)} for i in range(count)]
    if len({r['frame_index'] for r in rows})!=len(rows):raise ValueError('Interval produces repeated frame indices')
    if any(r['frame_index']>=frame_count for r in rows):raise ValueError('Sampling range extends beyond the last video frame')
    for r in rows:r['frame_seconds']=r['frame_index']/fps
    return rows
