"""
Main function.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import math
import time
import numpy as np

# Pytorch packages
import torch
import torch.optim as optim
import torch.nn as nn
# Torchtest packages
from torchtext.data import Field, LabelField, BucketIterator

from torch.utils.data import Dataset

from Transformer import Transformer

# Tqdm progress bar
from tqdm import tqdm_notebook

from torchtext import data


# Check device availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)
# Hyperparameters
BATCH_SIZE = 128
MAX_LEN = 472

# Define the source and target language
TEXT = Field(tokenize='spacy', batch_first=True, include_lengths=True)
LABEL = LabelField(dtype=torch.float, batch_first=True)
fields = [(None, None), ('context', TEXT),('label', LABEL)]

#loading custom dataset
train_data=data.TabularDataset(
    path='./data/train_captioned.csv', 
    format='csv',
    fields=fields,
    skip_header = True
    )

valid_data=data.TabularDataset(
    path='./data/dev_seen_captioned.csv', 
    format='csv',
    fields=fields,
    skip_header = True
    )

test_data=data.TabularDataset(
    path='./data/test_seen_captioned.csv', 
    format='csv',
    fields=fields,
    skip_header = True
    )

#print preprocessed text
print(vars(train_data.examples[0]))

"""
TEXT = Field(
    tokenize="spacy",
    tokenizer_language="en",
    init_token='<sos>',
    eos_token='<eos>',
    fix_length=MAX_LEN,
    lower=True
    )
"""

# Build the vocabulary
TEXT.build_vocab(train_data, min_freq=2)

train_loader, valid_loader, test_loader = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

# Get the input and the output sizes
input_size = len(TEXT.vocab)
output_size = 2

# Hyperparameters
learning_rate = 5

# Model
model = Transformer(input_size, output_size, device, max_length=MAX_LEN).to(device)

# optimizer = optim.Adam(model.parameters(), lr = learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# Ignore padding index when calculating cross entropy
criterion = nn.CrossEntropyLoss()


def train(model, dataloader, optimizer, criterion, scheduler=None):
    model.train()

    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii=True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        source = data.src.transpose(1, 0)
        target = data.trg.transpose(1, 0)

        translation = model(source)
        translation = translation.reshape(-1, translation.shape[-1])
        target = target.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(translation, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss
        progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar 
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            source = data.src.transpose(1, 0)
            target = data.trg.transpose(1, 0)

            translation = model(source)
            # translation = translation[:,1:].reshape(-1, translation.shape[-1])
            # target = target[1:].view(-1)
            translation = translation.reshape(-1, translation.shape[-1])
            target = target.reshape(-1)

            loss = criterion(translation, target)
            total_loss += loss
            progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss
    # print("Validation Loss: %.4f. Average Loss: %.4f. Perplexity" % (total_loss, avg_loss))


EPOCHS = 50
for epoch_idx in range(EPOCHS):
    print("-----------------------------------")
    print("Epoch %d" % (epoch_idx + 1))
    print("-----------------------------------")

    train_loss, avg_train_loss = train(model, train_loader, optimizer, criterion)
    scheduler.step(train_loss)

    val_loss, avg_val_loss = evaluate(model, valid_loader, criterion)

    avg_train_loss = avg_train_loss.item()
    avg_val_loss = avg_val_loss.item()
    print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
    print("Training Perplexity: %.4f. Validation Perplexity: %.4f. " % (np.exp(avg_train_loss), np.exp(avg_val_loss)))
