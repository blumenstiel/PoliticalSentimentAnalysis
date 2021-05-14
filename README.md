# PoliticalSentimentAnalysis
Analyzing Tweets of US Politicians with Sentiment Analysis

Author: Benedikt Blumenstiel

## Setup

Create a [miniconda](https://docs.conda.io/en/latest/miniconda.html) environment by running the following command:

```bash
conda env create -f environment.yml
```

## Data Collection
The scripts has been run in following order:
* src/data/get_tweets.py
* src/data/preprocessing.py
* src/data/label_emotion_scores.py
* src/data/label_sentiment_scores.py
* src/data/add_meta_data.py

To download the tweets using the Twitter API, keys and passwords of the developer account has to be set as a enviromental variable, e.g. in a extra script twitterkey.py.

## Experiments
Experiments to compare different classifiers:
* First, the BERT model was fine-tuned on a subset of s140 dataset: experiments/train_bert_classifier.py
* experiments/compare_sentiment_anaylser_s140.py
* experiments/compare_sentiment_anaylser_UScongress.py

The experiments include the Google Cloud Natural Language API. Therefore a Google Cloud account and project is required. The link to the google_api_credentials.json file can be specified in the scripts. 

The s140 dataset is provided here: https://www.kaggle.com/kazanova/sentiment140

The UScongress dataset was collected from Twitter on 13.04.2021 based on a list of all senators and representatives from https://ucsd.libguides.com/congress_twitter (data/congress_twitter_accounts.xlsx). 
Following the Developer Agreement if Twitter, only the tweet ids are shared (data/tweet_ids.csv). 

## Analysis
The analyis was performed in notebooks, available in the directory "notebooks".