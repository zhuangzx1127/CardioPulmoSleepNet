"""Verify the source/split release using only Python's standard library."""
from pathlib import Path
import ast,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for line in (ROOT/'BASELINES_SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1)
    path=ROOT/name
    if not path.is_file() or sha(path)!=expected:raise ValueError('Release checksum mismatch: '+name)
for name in ['oxinet','wav2sleep','bcgnet','driven']:
    base=ROOT/'Baselines'/name
    for item in json.loads((base/'upstream_inventory.json').read_text())['files']:
        if sha(base/'upstream'/item['path'])!=item['sha256']:raise ValueError('Upstream mismatch: '+name+'/'+item['path'])
for path in (ROOT/'Baselines').rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
documents=[json.loads((ROOT/p).read_text()) for p in ['Dataset/splits.json','BaselineInputs/native_10h/splits.json']]
for doc in documents:
    patients={p:set() for p in ['train','val','test']};seen=set()
    for cohort,parts in doc.items():
        for part,rows in parts.items():
            for row in rows:
                if row['recording_id'] in seen:raise ValueError('Repeated recording')
                seen.add(row['recording_id']);patients[part].add(row['participant_id'])
    if patients['train']&patients['val'] or patients['train']&patients['test'] or patients['val']&patients['test']:raise ValueError('Participant leakage')
for cohort in documents[0]:
    for part in ['train','val','test']:
        a={r['recording_id']:r['participant_id'] for r in documents[0][cohort][part]}
        b={r['recording_id']:r['participant_id'] for r in documents[1][cohort][part]}
        if not b.items()<=a.items() or (part!='train' and a!=b):raise ValueError('Native split mismatch')
print(json.dumps({'status':'PASS','upstream_inventories':4,'splits':2,'participant_disjoint':True,'native_val_test_equal_main':True}))
