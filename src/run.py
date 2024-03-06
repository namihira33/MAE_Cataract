from trainer import Trainer
import numpy as np
import sys

c = {'model_name': ['MAE_ViT'],'seed':[0], 'bs': 3}

args = len(sys.argv)
if args > 1:
    print(sys.argv)
    # c['model_name'] = sys.argv[1]
    c['cv'] = int(sys.argv[1].split('=')[1])
    c['evaluate'] = int(sys.argv[2].split('=')[1])
    c['d_mode'] = sys.argv[3].split('=')[1]
    c['type'] = sys.argv[4].split('=')[1]
    c['preprocess'] = sys.argv[5].split('=')[1]
    c['sampler'] = sys.argv[6].split('=')[1]
    c['gamma'] = float(sys.argv[7].split('=')[1])
    c['beta'] = float(sys.argv[8].split('=')[1])
    lr = float(sys.argv[9].split('=')[1])
    if lr > 0:
        c['lr'] = lr
    c['n_epoch'] = int(sys.argv[10].split('=')[1])

trainer = Trainer(c)
trainer.run()