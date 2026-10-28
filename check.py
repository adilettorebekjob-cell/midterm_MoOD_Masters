import json
from pathlib import Path
from prepare import fingerprint
root=Path(__file__).resolve().parent
sets=[]
for split in ['train','valid','test']:
    rows=[json.loads(s) for s in (root/'data'/f'{split}.jsonl').read_text().splitlines()]
    hashes={r['sha256'] for r in rows}
    assert len(hashes)==len(rows)
    assert {r['target'] for r in rows}=={0,1}
    assert all(fingerprint(r['code'])==r['sha256'] for r in rows)
    sets.append(hashes)
assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
print('PASS: nonempty code, both classes, unique hashes and no exact overlap across splits')
