"""Run from PyCharm, Windows terminal or notebook using the active interpreter."""
import subprocess,sys
from pathlib import Path
from datetime import datetime
import torch
root=Path(__file__).resolve().parent
if not (root/'data/train.jsonl').exists():
    raise RuntimeError('Copy the patch files into your original Midterm_Group3 folder, beside data and train.py')
if not torch.cuda.is_available():
    raise RuntimeError('GPU not visible in this interpreter. Select the venv with torch 2.8.0+cu128')
name='codet5_v2_'+datetime.now().strftime('%Y%m%d_%H%M%S')
(root/'results').mkdir(exist_ok=True)
print('Python:',sys.executable,'GPU:',torch.cuda.get_device_name(0),'Run:',name,flush=True)
with (root/'results'/f'{name}_console.log').open('x',encoding='utf-8') as log:
    process=subprocess.Popen([sys.executable,'-u',str(root/'train_v2.py'),'--output',name],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    try:
        for line in process.stdout:
            print(line,end='',flush=True); log.write(line);log.flush()
        result=process.wait()
    except KeyboardInterrupt:
        process.terminate();process.wait();raise
if result:raise RuntimeError(f'Training failed with exit code {result}; inspect the saved console log')
print('Finished. Results:',root/'results'/name)
