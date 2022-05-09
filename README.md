# Deep Learning (CS7643) Group Project
## Dan Tylutki, Hassan Naveed, Jake Banigan, Saeb Hashish
NOTE: This was a joint project forked from the team's github enterprise and made available publicly. The report is available as Project_Report__CS_7643.pdf

### Baseline Notebook:

Our baseline for experiments was generated in the Notebook title *Hateful_memes_group_Baseline.ipynb*. This is available as a google colab notebook here:
https://colab.research.google.com/drive/1nZqnmlYjudt2bOaiqCM2PzKss_CKtuPY?usp=sharing

#### Introduction
This notebook aims to establish a baseline we will be using for experiments. It will contain the following unimodally pretained components:
- BERT for text classification
- CLIP for visual embeddings

For the classification head, it will use default head of distilBERTforSequenceClassication

#### Loading Data:
Next we need to connect to the storage. The script assumes the following folder structure:
```
.
|-hateful_memes
  |-dev_seen.jsonl
  |-train_jsonl
  |-img
    |-{all memes}
|-embeds
  |-{all embeds}
```

So basically just make sure you have the unzipped hateful memes folder in the current working directory. Feel free to change this part to something that works with your environment

#### Embeddings:
The sections 1 of the notebook creates text embeddings, whereas section 2 creates vision embeddings. These embeddings will be saved on the device the notebook is run on. Some of these take time to generate, so its faster to save them for reuse later.

NOTE: 16gb of GPU memory is recommended for generating vision embeddings

### Classification Layer:

As mentioned earlier, the default head of the DistilBERTforSequenceCLassification is used, which comprises of two linear layers. The concatenated embeddings pass through the following layers:
- linear layer
- ReLU
- dropout
- linear layer
- sigmoid

The classification head was kept simple so valid comparisions between the importance of embeddings could be compared.
