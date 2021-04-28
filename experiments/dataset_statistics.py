
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import TweetTokenizer

# laod data
tweets_df = pd.read_pickle('../data/tweets_raw.pkl')

# add user information
user_df = pd.read_excel('../data/congress_twitter_accounts.xlsx')
user_df.set_index('user', inplace=True)
party_dict = user_df.party.to_dict()
tweets_df['party'] = tweets_df.user.apply(lambda u: party_dict[u])


# Simple counts
print(f'Number of tweets: {len(tweets_df)}')
print(f'Number of users: {len(tweets_df.user.unique())}')
print(f'Average number of tweets per user: {round(len(tweets_df) / len(tweets_df.user.unique()), 2)}')

# Histogram of num tweets per user and party
user_party_num = tweets_df.groupby('user')['id'].count()
user_party_num = user_party_num.drop('I').reset_index().set_index('user')
sns.displot(user_party_num, x='id', hue='party', binwidth=100, kde=True)
plt.xlabel('Number of tweets')
plt.ylabel('Number of users')
plt.show()

