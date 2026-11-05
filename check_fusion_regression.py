"""Read-only frozen development regression: validate result binding separately from recognition."""
import argparse,json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent

def check_case(case,result):
 actual=hashlib.sha256(Path(case['image']).read_bytes()).hexdigest()
 bound=actual==case['sha256']==result.get('sha256')
 functional=bound and result.get('status')=='needs_review' and result.get('compliance_determined') is False and result.get('lines',{}).get('vlm',{}).get('performed') is True and result['lines']['vlm'].get('valid') is True
 present=result.get('board_presence','').startswith('present')
 return {'id':case['id'],'binding_ok':bound,'functional_pass':functional,'recognition_pass':functional and present==case['expected_board_present'],'expected_board_present':case['expected_board_present'],'observed':result.get('board_presence')}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 cases=json.loads(args.manifest.read_text(encoding='utf-8'))['cases'];rows=[]
 for case in cases:rows.append(check_case(case,json.loads(Path(case['result']).read_text(encoding='utf-8'))))
 summary={'cases':rows,'functional_pass':all(r['functional_pass'] for r in rows),'recognition_pass':all(r['recognition_pass'] for r in rows),'scope':'Previously viewed development cases; cached inference, not a fresh model run'}
 args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False));return 0 if summary['recognition_pass'] else 1
if __name__=='__main__':raise SystemExit(main())
