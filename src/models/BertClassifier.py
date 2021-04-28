
# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
# https://www.tensorflow.org/tutorials/text/classify_text_with_bert
# https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671
# https://github.com/OthSay/bert-tweets-analysis
# https://github.com/akoksal/BERT-Sentiment-Analysis-Turkish/blob/master/BERT%20Features%20with%20Keras.ipynb

# -> https://www.kaggle.com/menion/sentiment-analysis-with-bert-87-accuracya

import os
import shutil
import numpy as np
import pandas as pd
from BertLibrary import BertFTModel
import re

_root_path = '../../'

# LOAD DATA
t140 = pd.read_csv(_root_path + 'data/sentiment140/training.1600000.processed.noemoticon.csv',
                   sep=',',
                   header=None,
                   encoding='latin')

label_text = t140[[0, 5]]

# Convert labels to range -1 to 1
label_text[0] = label_text[0].apply(lambda x: -1 if x == 0 else 1)

# Assign proper column names to labels
label_text.columns = ['label', 'text']

# Assign proper column names to labels
# label_text.head()

# PREPROCESSING


# regex for hastags, mentions and urls
hashtags = re.compile(r"^#\S+|\s#\S+")
mentions = re.compile(r"^@\S+|\s@\S+")
urls = re.compile(r"https?://\S+")
whitespaces = re.compile(r"\s\s+")
non_characters = re.compile(r"[^A-Za-z\s]+")
short_words = re.compile(r"^\S\S?\s|\s\S\S?\s|\s\S\S?$")


def preprocess(text):
    text = hashtags.sub('', text)
    text = mentions.sub('', text)
    text = urls.sub('', text)
    # remove all punctuation, numbers, emojis etc.
    text = non_characters.sub('', text)
    # double white spaces, spaces on beginning or ending and lower case
    text = whitespaces.sub(' ', text).strip().lower()
    return text


label_text.text = label_text.text.apply(preprocess)

# TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.75
VAL_SIZE = 0.05
dataset_count = len(label_text)

df_train_val, df_test = train_test_split(label_text, test_size=1 - TRAIN_SIZE - VAL_SIZE, random_state=42)
df_train, df_val = train_test_split(df_train_val, test_size=VAL_SIZE / (VAL_SIZE + TRAIN_SIZE), random_state=42)

print("TRAIN size:", len(df_train))
print("VAL size:", len(df_val))
print("TEST size:", len(df_test))

dataset_path = _root_path + 'data/sentiment140/dataset'
os.mkdir(dataset_path)
df_train.sample(frac=1.0).reset_index(drop=True).to_csv(dataset_path + '/train.tsv', sep='\t', index=None, header=None)
df_val.to_csv(dataset_path + '/dev.tsv', sep='\t', index=None, header=None)
df_test.to_csv(dataset_path + '/test.tsv', sep='\t', index=None, header=None)


os.system('wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')
os.system('unzip uncased_L-12_H-768_A-12.zip')

ft_model = BertFTModel(model_dir='uncased_L-12_H-768_A-12',
                       ckpt_name="bert_model.ckpt",
                       labels=['-1', '1'],
                       lr=1e-05,
                       num_train_steps=30000,
                       num_warmup_steps=1000,
                       ckpt_output_dir='output',
                       save_check_steps=1000,
                       do_lower_case=False,
                       max_seq_len=50,
                       batch_size=32,
                       )

ft_trainer =  ft_model.get_trainer()
ft_evaluator = ft_model.get_evaluator()
ft_trainer.train_from_file(dataset_path, 35000)
ft_evaluator.evaluate_from_file(dataset_path, checkpoint="output/model.ckpt-35000")