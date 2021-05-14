import pandas as pd
import pickle5 as pickle

_root_path = '../' # '/content/drive/MyDrive/PoliticalSentimentAnalysis/'

# load data
with open(_root_path + 'data/tweets_labeled.pkl', 'rb') as f:
    tweets_df = pickle.load(f)

# read user information
user_df = pd.read_excel(_root_path + 'data/congress_twitter_accounts.xlsx')
user_df.set_index('user', inplace=True)

# add party
party_dict = user_df.party.to_dict()
tweets_df['Party'] = tweets_df.user.apply(lambda u: party_dict[u])
# add state
state_dict = user_df.state.to_dict()
tweets_df['State'] = tweets_df.user.apply(lambda u: state_dict[u])
# add house
house_dict = user_df.house.to_dict()
tweets_df['House'] = tweets_df.user.apply(lambda u: house_dict[u])

# add quality gate for consistent labels between BERT and VADER
tweets_df['consistent_label'] = (tweets_df.label == tweets_df.vader_label)

# rename for clearer plots
tweets_df.rename(columns={
    'created_at': 'Date',
    'favorite_count': 'Likes',
    'retweet_count': 'Retweets'
}, inplace=True)

# save data
tweets_df.to_pickle(_root_path + 'data/tweets.pkl')
tweets_df.to_csv(_root_path + 'data/tweets.csv')

# export ids for review
tweets_df.id.to_csv(_root_path + 'data/tweet_ids.csv')