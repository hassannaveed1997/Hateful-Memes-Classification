import torch
import torch.nn as nn

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VL_VBert(nn.Module):
    def __init__(self, vis_mdl, lang_mdl, vbert, dp=0.25, n_out=1):
        super().__init__()
        self.vis_cfg = vis_mdl.config.to_dict()
        self.lang_cfg = lang_mdl.config.to_dict()
        self.vb_cfg = vbert.config.to_dict()
        self.vbert = vbert

        self.vis_mdl = vis_mdl
        self.lang_mdl = lang_mdl


        #n_emb = self.lang_cfg['hidden_size'] + self.vis_cfg['hidden_size']
        n_emb = self.vb_cfg['hidden_size']
        self.n_emb = n_emb
        n_emb_vis = self.vb_cfg['visual_embedding_dim']
        self.n_emb_vis = n_emb_vis

        self.drp = nn.Dropout(dp)
        #self.relu = nn.ReLU()
        #self.nrm = nn.LayerNorm(n_emb)
        #self.ln = nn.Linear(n_emb, n_emb)
        self.out = nn.Linear(n_emb, n_out)

    def get_hid(self, out, mode='sum'):
        hid = sum(out['hidden_states'][-4:])
        if mode == 'sum':
            hid = torch.sum(hid[:,:4,:], axis=1)
        if mode == 'first':
            hid = hid[:,0,:]
        return hid

    def vis_fwd(self, img, mode='sum'):
        out = self.vis_mdl(img, output_hidden_states=True)
        hid = self.get_hid(out, mode=mode)
        return hid

    def lang_fwd(self, txt, mode='sum'):
        out = self.lang_mdl(txt, output_hidden_states=True)
        hid = sum(out['hidden_states'][-4:])
        hid = self.get_hid(out, mode=mode)
        return hid

    def forward(self, batch):
        txt, img = batch['text'].to(dev), batch['img'].to(dev)
        vis_hidden = self.vis_fwd(img, mode='sum').unsqueeze(1)
        vis_hidden = vis_hidden.repeat(1, 1, 3)[:,:,:self.n_emb_vis]
        self.vis_hidden = vis_hidden

        lang_hidden = self.lang_fwd(txt, mode='none')
        self.lang_hidden = lang_hidden

        #comb = torch.concat([vis_hidden, lang_hidden], axis=1)
        #self.comb = comb
        #pre = self.drp(self.nrm(self.ln(comb)))
        pre = self.vbert(inputs_embeds=lang_hidden, visual_embeds=vis_hidden, output_hidden_states=True)
        pre = self.get_hid(pre, mode='first')
        logs = self.out(pre)
        return logs


class VLModel(nn.Module):
    def __init__(self, vis_mdl, lang_mdl, dp=0.25, n_out=1):
        super().__init__()
        self.vis_cfg = vis_mdl.config.to_dict()
        self.lang_cfg = lang_mdl.config.to_dict()
        self.vis_mdl = vis_mdl
        self.lang_mdl = lang_mdl
        n_emb = self.lang_cfg['hidden_size'] + self.vis_cfg['hidden_size']


        self.drp = nn.Dropout(dp)
        self.relu = nn.ReLU()
        self.nrm = nn.LayerNorm(n_emb)
        self.ln = nn.Linear(n_emb, n_emb)
        self.out = nn.Linear(n_emb, n_out)

    def get_hid(self, out, mode='sum'):
        hid = sum(out['hidden_states'][-4:])
        if mode == 'sum':
            hid = torch.sum(hid[:,:4,:], axis=1)
        if mode == 'first':
            hid = hid[:,0,:]
        return hid

    def vis_fwd(self, img, mode='sum'):
        out = self.vis_mdl(img, output_hidden_states=True)
        hid = self.get_hid(out, mode=mode)
        return hid

    def lang_fwd(self, txt, mode='sum'):
        out = self.lang_mdl(txt, output_hidden_states=True)
        hid = sum(out['hidden_states'][-4:])
        hid = self.get_hid(out, mode=mode)
        return hid

    def forward(self, batch):
        txt, img = batch['text'].to(dev), batch['img'].to(dev)
        vis_hidden = self.vis_fwd(img)
        self.vis_hidden = vis_hidden

        lang_hidden = self.lang_fwd(txt)
        self.lang_hidden = lang_hidden

        comb = torch.concat([vis_hidden, lang_hidden], axis=1)
        self.comb = comb
        pre = self.drp(self.relu(self.nrm(self.ln(comb))))
        logs = self.out(pre)
        return logs

