"""Classify a Java text file using a completed experiment's best checkpoint."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from train import CodeT5Classifier
p=argparse.ArgumentParser();p.add_argument('java_file');p.add_argument('--run',default='results/codet5_base');a=p.parse_args()
r=Path(a.run);cfg=json.loads((r/'config.json').read_text());device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m=CodeT5Classifier(cfg['model']).to(device);m.load_state_dict(torch.load(r/'best.pt',map_location=device,weights_only=True));m.eval()
tok=AutoTokenizer.from_pretrained(r/'tokenizer');x=tok(Path(a.java_file).read_text(),return_tensors='pt',truncation=True,max_length=cfg['max_length'])
with torch.no_grad():v=m(**{k:t.to(device) for k,t in x.items() if k in ['input_ids','attention_mask']}).softmax(-1)[0,1].item()
print(json.dumps({'probability_vulnerable':v,'predicted_label':int(v>=.5),'threshold':.5,'note':'Research model; manual security review required'}))
