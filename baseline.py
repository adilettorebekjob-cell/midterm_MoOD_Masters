"""Diagnostic baselines only; neither replaces the assigned CodeT5 method."""
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
    rows={s:[json.loads(x) for x in (ROOT/'data'/f'{s}.jsonl').read_text().splitlines()] for s in ['train','valid','test']}
    v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=30000,min_df=3,sublinear_tf=True)
    x=v.fit_transform(r['code'] for r in rows['train']); y=[r['target'] for r in rows['train']]
    clf=LogisticRegression(class_weight='balanced',max_iter=500,random_state=42).fit(x,y)
    result={}
    for s in ['valid','test']:
        y=[r['target'] for r in rows[s]]; p=clf.predict_proba(v.transform(r['code'] for r in rows[s]))[:,1]; pred=p>=.5
        result[s]={'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'roc_auc':roc_auc_score(y,p),'confusion_matrix':confusion_matrix(y,pred).tolist(),'majority_accuracy':float(np.mean(np.array(y)==0)),'majority_recall':0}
    (ROOT/'results/baseline.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
