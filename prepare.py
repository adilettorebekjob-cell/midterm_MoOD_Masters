"""Validate supplied splits, quarantine conflicting labels and remove cross-split duplicates."""
import json, hashlib, tarfile, urllib.request
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
URL='https://raw.githubusercontent.com/rmusab/vul-llm-finetune/main/Datasets/with_p3/java_k_1_strict_2023_06_30.tar.gz'
def fingerprint(code):
    return hashlib.sha256(' '.join(code.split()).encode()).hexdigest()
def main():
    data=ROOT/'data'; data.mkdir(exist_ok=True); (ROOT/'results').mkdir(exist_ok=True)
    archive=data/'java.tar.gz'
    if not archive.exists(): urllib.request.urlretrieve(URL,archive)
    rows={}; labels={}; audit={'source':URL,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    with tarfile.open(archive) as tar:
        for split in ['train','valid','test']:
            rows[split]=[]; invalid=0
            for line in tar.extractfile(split+'.jsonl'):
                r=json.loads(line)
                if not isinstance(r.get('code'),str) or not r['code'].strip() or type(r.get('target')) is not int or r['target'] not in (0,1): invalid+=1; continue
                r['sha256']=fingerprint(r['code']); labels.setdefault(r['sha256'],set()).add(r['target']); rows[split].append(r)
            audit[split]={'raw_valid_rows':len(rows[split]),'invalid_rows':invalid}
    conflict={h for h,v in labels.items() if len(v)>1}; seen=set()
    # Preserve original test partition. Remove overlapping hashes from train/validation.
    for split in ['test','valid','train']:
        kept=[]; duplicates=0; conflicts=0
        for r in rows[split]:
            h=r['sha256']
            if h in conflict: conflicts+=1; continue
            if h in seen: duplicates+=1; continue
            seen.add(h); kept.append(r)
        audit[split].update(clean_rows=len(kept),duplicates_removed=duplicates,conflicts_removed=conflicts,classes=dict(Counter(str(r['target']) for r in kept)))
        with (data/(split+'.jsonl')).open('w') as f:
            for r in kept: f.write(json.dumps(r)+'\n')
    audit['conflicting_hashes']=len(conflict)
    (ROOT/'results/data_quality.json').write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
