# coding: utf-8

import torch
import joblib
import json
import subprocess
import pandas as pd
from bertqa_sklearn_fr import BertProcessor, BertQA
import re, os

input_file = './data/SQuAD_FR/annotations-18112019.json'

with open(input_file) as json_file:
    d = json.load(json_file)

print(len(d))

train_processor = BertProcessor(bert_model='bert-base-multilingual-uncased', do_lower_case=True, is_training=True)
train_examples, train_features = train_processor.fit_transform(X=input_file)

reader = BertQA(train_batch_size=12,
                learning_rate=3e-5,
                num_train_epochs=2,
                do_lower_case=True,
                output_dir='models')

reader.model.to('cpu')
reader.device = torch.device('cpu')

reader.fit(X=(train_examples, train_features))

joblib.dump(reader, os.path.join(reader.output_dir, 'bert_qa_fr.joblib'))
