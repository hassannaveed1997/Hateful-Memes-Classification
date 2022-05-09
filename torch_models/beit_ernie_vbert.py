import random
import numpy as np
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import n_param, param_nm, frz, get_tokid, acc, train, eval, train_loop
from loaders import VLData
from models import VL_VBert

from transformers import optimization as hoptim
from transformers import VisualBertModel, VisualBertConfig
from transformers import BertTokenizer, BertModel
from transformers import BeitFeatureExtractor, BeitModel


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('dev:', dev)
max_len = 400

tokr = BertTokenizer.from_pretrained('nghuyong/ernie-2.0-en')
lang_model = BertModel.from_pretrained('nghuyong/ernie-2.0-en')

vis_ftm = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k')
vision_model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k')

vb_cfg = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre", num_hidden_layers=2)
vbert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre", config=vb_cfg)


img_tfm = lambda im: vis_ftm(images=im, return_tensors='pt', do_center_crop=False)['pixel_values'][0]
txt_tfm = lambda txt: get_tokid(txt, tokr, max_len)


fls = ['train', 'dev_seen', 'test_seen']
#fl_nms = [fl+'.jsonl' for fl in fls]
fl_nms = [fl+'_captioned.csv' for fl in fls]

root, data_dir = '.', 'hateful_memes'
dr = join(root, data_dir)

fl_pts = [join(dr, nm) for nm in fl_nms]

b_sz = 128
loaders = []
for pt in fl_pts:
    data = VLData(pt, dr, img_tfm, txt_tfm)
    ldr = DataLoader(data, shuffle=True, batch_size=b_sz, num_workers=6)
    loaders.append(ldr)

trn_gen, vld_gen, tst_gen = loaders



#frz(vbert, 'embed')
print('params:', n_param(vbert))
lang_model.to(dev)
vision_model.to(dev)
vbert.to(dev)
model = VL_VBert(vision_model, lang_model, vbert)
print('params:', n_param(model))
frz(lang_model, 'embed')
frz(lang_model, 'encod')
frz(lang_model, 'pool')
print('params:', n_param(model))
frz(vision_model, 'embed')
frz(vision_model, 'encod')
frz(vision_model, 'pool')
print('params:', n_param(model))


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', dest='lr', default=2e-5, type=float)
parser.add_argument('--epochs', dest='ep', default=24, type=int)
args = parser.parse_args()
lr = args.lr
n_epoch = args.ep
print('using lr:', lr, 'epochs: ', n_epoch)

#opt = optim.AdamW(model.parameters(), lr=1e-5)
#opt = hoptim.AdamW(model.parameters(), lr=1e-5, correct_bias=False)
opt = hoptim.AdamW(model.parameters(), lr=lr)

#sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=2)
sched = optim.lr_scheduler.StepLR(opt, step_size=int(n_epoch/4), gamma=0.75)

loss = nn.BCEWithLogitsLoss()
model = model.to(dev)
loss = loss.to(dev)

cpt_nm = 'beit_ernie_vbert'

train_loop(model, trn_gen, vld_gen, tst_gen, opt, loss, sched=sched, cpt_nm=cpt_nm, n_epoch=n_epoch)
