import numpy as np
import torch
from unittest.mock import patch
from transformers import T5Config,T5Model
from train_v2 import select_threshold,metrics,CodeT5Classifier
# Scores entirely below .5 must still produce a useful validation threshold.
y=[0,0,1,1]; p=[.001,.002,.03,.04]
t=select_threshold(y,p)
assert t==.03 and metrics(y,p,t)['f1']==1 and metrics(y,p)['recall']==0
# Ties and duplicated probability scores use the same >= rule in selection and evaluation.
y=[0,1,0,1];p=[.01,.01,.04,.08]
t=select_threshold(y,p)
assert metrics(y,p,t)['f1']==max(metrics(y,p,x)['f1'] for x in set(p))
# Weighting must distinguish errors on rare positives, including single-class batches.
w=torch.tensor([.5155,16.6654]); loss=torch.nn.CrossEntropyLoss(weight=w,reduction='none')
a=loss(torch.tensor([[0.,0.]]),torch.tensor([0])).mean()
b=loss(torch.tensor([[0.,0.]]),torch.tensor([1])).mean(); assert b>30*a
cfg=T5Config(vocab_size=64,d_model=16,d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,pad_token_id=0,eos_token_id=2)
with patch('train_v2.T5Model.from_pretrained',return_value=T5Model(cfg)):
 m=CodeT5Classifier('tiny-test')
x=torch.tensor([[1,9,2,0],[1,8,7,2]])
z=m(x,x.ne(0).long());loss(z,torch.tensor([0,1])).mean().backward();assert m.head[-1].weight.grad is not None
print('PASS: validation threshold, tied scores, global class weighting, model forward/backward')
# Entire two-epoch pipeline in a temporary directory, using a tiny random backbone.
import tempfile,json,sys
from pathlib import Path
import train_v2
class TinyTokenizer:
    def __call__(self,code,**kw):
        ids=torch.tensor([[1,7 if code=='positive' else 9,2,0]])
        return {'input_ids':ids,'attention_mask':ids.ne(0).long()}
    def save_pretrained(self,path):
        Path(path).mkdir(exist_ok=True)
with tempfile.TemporaryDirectory() as td:
    root=Path(td);(root/'data').mkdir()
    rows=[{'code':'positive' if i%2 else 'negative','target':i%2,'sha256':str(i)} for i in range(8)]
    for split in ['train','valid','test']:
        (root/'data'/f'{split}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    with patch.object(train_v2,'ROOT',root),patch.object(sys,'argv',['train_v2.py','--epochs','2','--batch-size','2','--max-length','4','--output','integration']),patch('train_v2.AutoTokenizer.from_pretrained',return_value=TinyTokenizer()),patch('train_v2.T5Model.from_pretrained',return_value=T5Model(cfg)):
        train_v2.main()
    out=root/'results/integration';c=json.loads((out/'config.json').read_text());sel=json.loads((out/'selection.json').read_text());m=json.loads((out/'metrics.json').read_text())
    assert c['status']=='complete' and c['threshold']==sel['threshold']==m['threshold']
    assert (out/'validation_predictions.jsonl').exists() and (out/'metrics_at_0_5.json').exists()
    assert len(json.loads((out/'history.json').read_text()))==2
print('PASS: end-to-end two-epoch pipeline, checkpoint reload, threshold persistence and plots (tiny random model only)')
