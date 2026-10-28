"""CodeT5 encoder-decoder with EOS classification head; validation selects checkpoint."""
import argparse,json,random,time,platform,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,T5Model
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
class CodeT5Classifier(nn.Module):
    def __init__(self,name):
        super().__init__(); self.backbone=T5Model.from_pretrained(name)
        d=self.backbone.config.d_model
        self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(d,d),nn.Tanh(),nn.Dropout(.1),nn.Linear(d,2))
    def forward(self,input_ids,attention_mask):
        # Mirror T5 sequence classification: shifted source in decoder; last EOS representation.
        decoder=self.backbone._shift_right(input_ids)
        h=self.backbone(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=decoder,use_cache=False).last_hidden_state
        mask=input_ids.eq(self.backbone.config.eos_token_id)
        pos=torch.arange(input_ids.shape[1],device=h.device).expand_as(input_ids).masked_fill(~mask,-1).max(1).values
        if (pos<0).any(): raise ValueError('Every input must contain EOS')
        return self.head(h[torch.arange(h.shape[0],device=h.device),pos])
class JavaData(Dataset):
    def __init__(self,split,tokenizer,length,limit=0):
        rows=[json.loads(x) for x in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]
        self.rows=rows[:limit] if limit else rows; self.tok=tokenizer; self.length=length
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; x=self.tok(r['code'],max_length=self.length,truncation=True,padding='max_length',return_tensors='pt')
        return {k:v.squeeze(0) for k,v in x.items() if k in ['input_ids','attention_mask']},r['target']
def metrics(y,p):
    pred=(np.array(p)>=.5).astype(int)
    return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),roc_auc=roc_auc_score(y,p) if len(set(y))==2 else None,confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist())
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); y=[]; p=[]
    for x,t in loader:
        logits=model(**{k:v.to(device) for k,v in x.items()}); p.extend(logits.softmax(-1)[:,1].cpu().tolist()); y.extend(t.tolist())
    return y,p,metrics(y,p)
def main():
    a=argparse.ArgumentParser(); a.add_argument('--model',default='Salesforce/codet5-base'); a.add_argument('--epochs',type=int,default=3); a.add_argument('--batch-size',type=int,default=4); a.add_argument('--accumulation',type=int,default=4); a.add_argument('--max-length',type=int,default=256); a.add_argument('--lr',type=float,default=2e-5); a.add_argument('--seed',type=int,default=42); a.add_argument('--limit',type=int,default=0); a.add_argument('--output',default='codet5'); args=a.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out=ROOT/'results'/args.output
    out.mkdir(parents=True,exist_ok=False) # Never overwrite previous experimental evidence.
    cfg=vars(args)|{'device':str(device),'torch':torch.__version__,'python':platform.python_version(),'status':'running','smoke_test_only':bool(args.limit)}
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    tok=AutoTokenizer.from_pretrained(args.model); model=CodeT5Classifier(args.model).to(device)
    cfg['model_revision']=getattr(model.backbone.config,'_commit_hash',None)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    loaders={s:DataLoader(JavaData(s,tok,args.max_length,args.limit),batch_size=args.batch_size,shuffle=s=='train') for s in ['train','valid','test']}
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01); criterion=nn.CrossEntropyLoss(); best=-1; history=[]
    start=time.time()
    for epoch in range(args.epochs):
        model.train(); opt.zero_grad(); total=0
        for step,(x,y) in enumerate(loaders['train']):
            loss=criterion(model(**{k:v.to(device) for k,v in x.items()}),y.to(device)); total+=loss.item()
            # Normalize the final, possibly short accumulation group correctly.
            n=len(loaders['train']); group_size=min(args.accumulation,n-(step//args.accumulation)*args.accumulation)
            (loss/group_size).backward()
            if (step+1)%args.accumulation==0 or step+1==n:
                nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad()
        _,_,m=evaluate(model,loaders['valid'],device); row={'epoch':epoch+1,'loss':total/len(loaders['train']),'validation':m,'elapsed_seconds':time.time()-start}; history.append(row)
        print(json.dumps(row),flush=True); (out/'history.json').write_text(json.dumps(history,indent=2))
        if m['f1']>best: best=m['f1']; torch.save(model.state_dict(),out/'best.pt'); tok.save_pretrained(out/'tokenizer')
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True)); y,p,m=evaluate(model,loaders['test'],device)
    m['accuracy_requirement_met']=None if args.limit else m['accuracy']>=.93; m['status']='SMOKE TEST ONLY' if args.limit else 'held-out evaluation'; (out/'metrics.json').write_text(json.dumps(m,indent=2))
    rows=loaders['test'].dataset.rows
    (out/'predictions.jsonl').write_text('\n'.join(json.dumps({'sha256':r['sha256'],'target':t,'probability':v}) for r,t,v in zip(rows,y,p)))
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(m['confusion_matrix'],cmap='Blues'); ax.set(xlabel='Predicted',ylabel='Actual',xticks=[0,1],yticks=[0,1],title='Held-out confusion matrix')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(m['confusion_matrix'][i][j]),ha='center')
    fig.tight_layout(); fig.savefig(out/'confusion_matrix.png'); plt.close(fig)
    if len(set(y))==2:
        fpr,tpr,_=roc_curve(y,p); plt.plot(fpr,tpr,label=f"AUC {m['roc_auc']:.3f}"); plt.plot([0,1],[0,1],'--'); plt.xlabel('False positive rate'); plt.ylabel('Recall'); plt.legend(); plt.tight_layout(); plt.savefig(out/'roc.png'); plt.close()
    cfg['status']='complete'; cfg['elapsed_seconds']=time.time()-start; (out/'config.json').write_text(json.dumps(cfg,indent=2)); print(json.dumps(m,indent=2))
if __name__=='__main__': main()
