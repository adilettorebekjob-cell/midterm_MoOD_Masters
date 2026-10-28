"""Offline architecture check with a tiny RANDOM T5; not a pretrained experiment."""
from unittest.mock import patch
import torch
from transformers import T5Config,T5Model
from train import CodeT5Classifier,metrics
cfg=T5Config(vocab_size=64,d_model=16,d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,pad_token_id=0,eos_token_id=2)
with patch('train.T5Model.from_pretrained',return_value=T5Model(cfg)):
    m=CodeT5Classifier('offline-test')
x=torch.tensor([[1,9,2,0],[1,8,7,2]])
y=m(x,x.ne(0).long()); assert y.shape==(2,2); loss=torch.nn.functional.cross_entropy(y,torch.tensor([0,1]));loss.backward()
assert m.head[-1].weight.grad is not None
assert m.backbone.encoder.block[0].layer[0].SelfAttention.q.weight.grad is not None
assert metrics([0,1],[.1,.9])['f1']==1
print('PASS: tiny random encoder-decoder forward/backward, head gradients, metrics; NOT pretrained training')
