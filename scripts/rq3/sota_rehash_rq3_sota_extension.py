from pathlib import Path
import hashlib
EXT=Path(__file__).resolve().parents[1];P=EXT/'protocol'
include=['scripts','configs','models','predictions','per_seed_results','tables','figures','model_reference_audit.md','adaptation_log.md','backbone_adaptation.md','RQ3_SOTA_EXTENSION_REPORT.md','FINALIZATION_STATUS.md','protocol/RQ3_SOTA_EXTENSION_FREEZE_v1.md']
files=[]
for item in include:
 p=EXT/item
 if p.is_file():files.append(p)
 elif p.exists():files.extend(q for q in p.rglob('*') if q.is_file())
files=[q for q in files if 'invalid_runs' not in q.parts and q.name!='RQ3_SOTA_EXTENSION_FREEZE_SHA256.txt']
lines=[]
for q in sorted(set(files)):lines.append(f'{hashlib.sha256(q.read_bytes()).hexdigest()}  {q.relative_to(EXT).as_posix()}')
(P/'RQ3_SOTA_EXTENSION_FREEZE_SHA256.txt').write_text('\n'.join(lines)+'\n');print('REHASH_PASS',len(lines))
