#!/usr/bin/env python
# coding: utf-8

# # Training the reader on SQuAD FR dataset

# This notebook shows how to fine-tune a pre-trained BERT model on the SQuAD.

# ***Original CDQA Note:*** *To run this notebook you will need to have access to GPU. The fine-tuning of the Reader was done with an AWS EC2 p3.2xlarge machine (GPU Tesla V100 16GB). It took about 2 hours to complete (2 epochs on SQuAD 1.1 train was enough to achieve SOTA results on SQuAD 1.1 dev).*

# In[1]:


import torch

import joblib
import json
import subprocess
import pandas as pd
from bertqa_sklearn_fr import BertProcessor, BertQA
import re, os


# ### Check SQuAD FR dataset

# In[2]:


input_file = './data/SQuAD_FR/annotations-24022020.json'


# In[3]:


with open(input_file) as json_file:
    d = json.load(json_file)


# In[4]:


# d[0]['paragraphs'][0]['questions']


# In[5]:


print(len(d['data']))


# ### Preprocess SQuAD examples

# In[6]:


train_processor = BertProcessor(bert_model='bert-base-uncased', do_lower_case=True, is_training=True)
train_examples, train_features = train_processor.fit_transform(X=input_file)


# ### Train the model

# In[ ]:





# In[7]:


reader = BertQA(train_batch_size=6,
                learning_rate=3e-5,
                num_train_epochs=10,
                do_lower_case=True,
                output_dir='models')


# In[8]:


# My GPU doesn't have engough memory (total 2GB), but comment this to use GPU instead of CPU
reader.model.to('cpu')
reader.device = torch.device('cpu')


# In[ ]:


reader.fit(X=(train_examples, train_features))


# ### Save model locally

# In[ ]:


joblib.dump(reader, os.path.join(reader.output_dir, 'bert_qa_fr.joblib'))


# In[ ]:




