import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from os.path import join


class VLData(Dataset):
    def __init__(self, dt_dir, img_dir, img_tfm, txt_tfm):
        if 'json' in dt_dir:
            self.dta = pd.read_json(dt_dir, lines=True)

        if 'csv' in dt_dir:
            self.dta = pd.read_csv(dt_dir)

        self.img_tfm = img_tfm
        self.txt_tfm = txt_tfm

        self.dta.img = self.dta.apply(lambda s: join(img_dir, s.img), axis=1)

    def __len__(self):
        return len(self.dta)

    def __getitem__(self, ix):

        if torch.is_tensor(ix):
            ix = ix.tolist()

        sid = self.dta.loc[ix, 'id']
        txt = self.txt_tfm(self.dta.loc[ix, 'text'])
        #print(txt)
        #print(tokenizer.convert_ids_to_tokens(txt))

        sample = {
            'id': sid,
            'text': txt,
            }

        if 'img' in self.dta.columns:
            img = Image.open(self.dta.loc[ix, 'img']).convert('RGB')
            img = self.img_tfm(img)
            #print(img)
            sample['img'] = img

        if 'caption' in self.dta.columns:
            txt = self.dta.loc[ix, 'text']
            cpt = self.dta.loc[ix, 'caption']
            txt = ' '.join([txt, cpt])
            txt = self.txt_tfm(txt)
            sample['text'] = txt
            
        if 'sent_pred' in self.dta.columns:
            pred = self.dta.loc[ix, 'sent_pred']
            prob = self.dta.loc[ix, 'sent_prob']
            sample['sent_pred'] = pred
            sample['sent_prob'] = prob

        if 'label' in self.dta.columns:
            lbl = self.dta.loc[ix, 'label']
            lbl = torch.tensor(lbl).long().squeeze()
            sample['label'] = lbl

        return sample

