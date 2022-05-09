import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel
from torchtext.legacy import data
from torchtext.legacy import datasets


seed = 132

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def get_tok(snt):
    tok = tokenizer.tokenize(snt)[:max_len-2]
    return tok

def get_tokid(snt):
    tok = tokenizer.encode_plus(
        snt,
        padding='max_length',
        max_length=max_len,
        add_special_tokens=True, 
        return_token_type_ids=False,
        return_attention_mask=False,
        return_tensors='pt',
        )

    tok = tok['input_ids'][0]

    return tok


b_sz = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class bert_gru(nn.Module):
    def __init__(self,
                 bert,
                 n_hidden,
                 n_out,
                 n_lyr,
                 dp):
        
        super().__init__()

        self.bert = bert
        n_emb = bert.config.to_dict()['hidden_size']
        
        self.gru = nn.GRU(
                    n_emb,
                    n_hidden,
                    num_layers=n_lyr,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dp
                    )
        
        self.lin = nn.Linear(n_hidden*2, n_out)
        
        self.drp = nn.Dropout(dp)
        
    def forward(self, txt):
        with torch.no_grad():
            hidden = self.bert(txt)[0]
        
        _, hidden = self.gru(hidden)
        hidden = self.drp(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        self.hidden = hidden
        out = self.lin(hidden)
        
        return out

def n_param(mdl):
    n_prm = sum(
            prm.numel() for prm in mdl.parameters()
                if prm.requires_grad
            )
    return n_prm

def frz_bert(mdl):
    for n,p in mdl.named_parameters():                
        if n.startswith('bert'): p.requires_grad = False

def acc(prob, y):
    y_hat = torch.round(torch.sigmoid(prob))
    corr = (y_hat == y).float()
    acc = corr.sum()/len(corr)
    return acc


def train(mdl, gen, opt, loss_fn):
    tot_loss = 0
    tot_acc = 0
    
    mdl.train()
    
    for bt in gen:
        opt.zero_grad()
        logs = mdl(bt.text).squeeze(1)
        loss = loss_fn(logs, bt.label)
        ac = acc(logs, bt.label)

        loss.backward()
        opt.step()
        
        tot_loss += loss.item()
        tot_acc += ac.item()
        
        mean_loss = tot_loss/len(gen)
        mean_acc = tot_acc/len(gen) 
        
    return mean_loss, mean_acc 


def eval(mdl, gen, loss_fn):
    
    tot_loss = 0
    tot_acc = 0
    mdl.eval()
    
    with torch.no_grad():
        for bt in gen:
            logs = mdl(bt.text).squeeze(1)
            loss = loss_fn(logs, bt.label)
            
            ac = acc(logs, bt.label)

            tot_loss += loss.item()
            tot_acc += ac.item()
        
        mean_loss = tot_loss/len(gen)
        mean_acc = tot_acc/len(gen) 
        
    return mean_loss, mean_acc 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--predict', dest='pred_md', default=False, action='store_true')
args = parser.parse_args()

pred_md = args.pred_md
trn_md = not pred_md

if trn_md:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    init_ix = tokenizer.cls_token_id
    eos_ix = tokenizer.sep_token_id
    pad_ix = tokenizer.pad_token_id
    unk_ix = tokenizer.unk_token_id

    max_len = tokenizer.max_model_input_sizes['bert-base-uncased']

    txt_fld = data.Field(
                  batch_first = True,
                  use_vocab = False,
                  tokenize = get_tok,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_ix,
                  eos_token = eos_ix,
                  pad_token = pad_ix,
                  unk_token = unk_ix
                  )

    lbl_fld = data.LabelField(dtype=torch.float)

    trn, tst = datasets.IMDB.splits(txt_fld, lbl_fld)
    trn, vld = trn.split(random_state=random.seed(seed))

    lbl_fld.build_vocab(trn)
    print(lbl_fld.vocab.stoi)

    print('trn:', len(trn), 'vld:', len(vld), 'tst:', len(tst))
    print(vars(trn.examples[6]))
    tok = tokenizer.convert_ids_to_tokens(vars(trn.examples[6])['text'])
    print(tok)

    trn_gen, vld_gen, tst_gen = data.BucketIterator.splits(
                    (trn, vld, tst), 
                    batch_size=b_sz, 
                    device=device
                    )

    n_hid = 256
    n_out = 1
    n_lyr = 2
    drop = 0.25

    bert = BertModel.from_pretrained('bert-base-uncased')
    model = bert_gru(bert,
             n_hid,
             n_out,
             n_lyr,
             drop
             )



    print('params:', n_param(model))
    frz_bert(model)
    print('params frozen:', n_param(model))


    opt = optim.Adam(model.parameters())
    loss = nn.BCEWithLogitsLoss()
    model = model.to(device)
    loss = loss.to(device)

    n_epoch = 6
    best_loss = np.inf

    for ep in range(n_epoch):
        
        start = time.time()
        
        trn_loss, trn_acc = train(model, trn_gen, opt, loss)
        vld_loss, vld_acc = eval(model, vld_gen, loss)
            
        end = time.time()
            
        ep_time = end - start
            
        if vld_loss < best_loss:
            best_loss = vld_loss
            torch.save(model.state_dict(), 'sent_model.pt')
        
        print('Epoch:', ep+1, '| Epoch Time:', ep_time)
        print('Train Loss: %.3f' % trn_loss, '| Train Acc: %.2f' % (trn_acc*100))
        print('Vld Loss: %.3f' % vld_loss, '| Vld Acc: %.2f' % (vld_acc*100))

    model.load_state_dict(torch.load('sent_model.pt'))
    tst_loss, tst_acc = eval(model, tst_gen, loss)
    print('Test Loss: %.3f' % tst_loss, '| Test Acc: %.2f' % (tst_acc*100))

elif pred_md:
    print('prediction route')

    from os.path import join as pjoin
    from os.path import sep 
    from PIL import Image
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from torch.utils.data import Dataset, DataLoader
    import pandas as pd

    def join(*args, **kwargs):
      return pjoin(*args, **kwargs).replace(sep, '/')


    img_sz = 224
    b_sz = 256

    img_tfm = Compose([
        Resize(size=(img_sz, img_sz)), 
        ToTensor(),
        Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
            )
        ])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = tokenizer.max_model_input_sizes['bert-base-uncased']
    txt_tfm = get_tokid

    class VLData(Dataset):
        def __init__(self, dt_dir, img_dir, img_tfm, txt_tfm):
            self.dta = pd.read_json(dt_dir, lines=True)
            self.img_tfm = img_tfm
            self.txt_tfm = txt_tfm

            self.dta.img = self.dta.apply(lambda s: join(img_dir, s.img), axis=1)

        def __len__(self):
            return len(self.dta)

        def __getitem__(self, ix):

            if torch.is_tensor(ix):
                ix = ix.tolist()

            sid = self.dta.loc[ix, 'id']
            img = Image.open(self.dta.loc[ix, 'img']).convert('RGB')
            img = self.img_tfm(img)
            txt = self.txt_tfm(self.dta.loc[ix, 'text'])
            #print(txt)
            #print(tokenizer.convert_ids_to_tokens(txt))
            #txt = torch.tensor(txt).squeeze()

            sample = {
                'id': sid,
                'image': img,
                'text': txt,
                }

            if 'label' in self.dta.columns:
                lbl = self.dta.loc[ix, 'label']
                lbl = torch.tensor(lbl).long().squeeze()
                sample['label'] = lbl

            return sample

    n_hid = 256
    n_out = 1
    n_lyr = 2
    drop = 0.25

    bert = BertModel.from_pretrained('bert-base-uncased')
    model = bert_gru(bert,
             n_hid,
             n_out,
             n_lyr,
             drop
             )
    model.load_state_dict(torch.load('sent_model.pt', map_location=device))
    model.to(device)

    fls = ['dev_seen', 'dev_unseen', 'test_seen', 'test_unseen', 'train']
    for fl in fls:
        fl_nm = fl+'.jsonl'
        root, data_dir = '.', 'hateful_memes'
        dr = join(root, data_dir)
        fl_pt = join(dr, fl_nm)

        data = VLData(fl_pt, dr, img_tfm, txt_tfm)
        loader = DataLoader(data, shuffle=True, batch_size=b_sz, num_workers=0)



        ids_lst = []
        probs_lst = []
        preds_lst = []
        hid_lst = []

        for i, s in enumerate(loader):
            txt = s['text']
            txt = txt.to(device)
            with torch.no_grad():
                logs = model.forward(txt)
                hid = model.hidden
                probs = torch.sigmoid(logs)
                preds = torch.round(probs)

            print(i, s['id'].shape, s['image'].shape, s['text'].shape, s['label'].shape)
            print('prob n', probs.shape)
            print('preds n', preds.shape)
            print('hid n', hid.shape)

            ids = s['id']
            ids_lst.append(ids.detach().cpu().numpy())
            probs_lst.append(probs.detach().cpu().numpy())
            preds_lst.append(preds.detach().cpu().numpy())
            hid_lst.append(hid.detach().cpu().numpy())

            #if i == 2:
            #    break


        ids = np.concatenate(ids_lst, axis=0).reshape(-1, 1)
        probs = np.concatenate(probs_lst, axis=0).reshape(-1, 1)
        preds = np.concatenate(preds_lst, axis=0).reshape(-1, 1)
        hid = np.concatenate(hid_lst, axis=0)
        print('hid lst', hid.shape)

        df = pd.DataFrame(np.concatenate([ids, probs, preds], axis=1), columns=['id', 'sent_prob', 'sent_pred'])

        df.to_csv(fl+'_sent.csv')
        np.savez(fl+'_sent.npz', hid)

