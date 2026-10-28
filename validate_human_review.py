"""Check exported notes against exact evidence; never accept notes as training truth."""
import argparse,hashlib,json
from pathlib import Path
DECISIONS={'pending','observation_supported','false_region','uncertain'}
def validate(review,events,source_hash):
 if review.get('source_events_sha256')!=source_hash:raise ValueError('Evidence SHA mismatch')
 rows=review.get('records');expected=len(events['records'])
 if not isinstance(rows,list) or len(rows)!=expected:raise ValueError('Review count mismatch')
 ids=[]
 for r in rows:
  if type(r.get('index')) is not int:raise ValueError('Invalid index')
  ids.append(r['index'])
  if r.get('decision') not in DECISIONS or not isinstance(r.get('note'),str):raise ValueError('Invalid review')
 if sorted(ids)!=list(range(expected)):raise ValueError('Missing or duplicate index')
 return {d:sum(r['decision']==d for r in rows) for d in sorted(DECISIONS)}
def main():
 p=argparse.ArgumentParser();p.add_argument('review',type=Path);p.add_argument('events',type=Path);a=p.parse_args();print(validate(json.loads(a.review.read_text(encoding='utf-8-sig')),json.loads(a.events.read_text(encoding='utf-8')),hashlib.sha256(a.events.read_bytes()).hexdigest()))
if __name__=='__main__':main()
