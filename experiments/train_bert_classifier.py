import pandas as pd
from src.models.BertClassifier import BertClassifier, BertDataset
from src.utils import sample_dataset
from sklearn.model_selection import train_test_split


_root_path = '/content/drive/MyDrive/PoliticalSentimentAnalysis/'

# LOAD DATA
t140 = pd.read_csv(_root_path + 'data/sentiment140/training.1600000.processed.noemoticon.csv',
                   sep=',',
                   header=None,
                   encoding='latin')

label_text = t140[[0, 5]]

# Convert labels to range -1 to 1
label_text[0] = label_text[0].apply(lambda x: 0 if x == 0 else 1)

# Assign proper column names to labels
label_text.columns = ['label', 'text']
label_text.reset_index(drop=True, inplace=True)

# init train and test set
num_test_tweets = 10000
test_df = sample_dataset(label_text, 'label', num_test_tweets)

# BERT
train_df = label_text.drop(test_df.index)
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_df.text.values,
                                                                                    train_df.label.values,
                                                                                    random_state=42,
                                                                                    test_size=0.1)
train_loader = BertDataset(train_inputs, train_labels).dataloader
val_loader = BertDataset(validation_inputs, validation_labels).dataloader
bert = BertClassifier()

bert.do_train(1, train_loader, val_loader, save_path=_root_path + 'models/bert/')