from tweepy import OAuthHandler, API, error
import pandas as pd
import os
import time


def get_twitter_api():
    # twitterkey.py is saving personal twitter keys as env variable
    import src.twitterkey

    # get api keys
    consumer_key = os.getenv('USER_KEY')
    consumer_secret = os.getenv('USER_SECRET')
    access_token = os.getenv('ACCESS_TOKEN')
    access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

    # Setup Tweepy
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = API(auth)

    return api


def get_user_tweets(api, users: list, save_path: str = None) -> pd.DataFrame:
    """Extract all tweets from users"""
    # code based on https://fairyonice.github.io/extract-someones-tweet-using-tweepy.html

    # initializing DataFrame (from saved pickle or empty)
    if os.path.isfile(save_path + '.pkl'):
        tweets_df = pd.read_pickle(save_path + '.pkl')
        # remove already downloaded users from list
        for user in tweets_df['user'].unique():
            users.remove(user)
    else:
        tweets_df = pd.DataFrame()

    print(f'Downloading tweets from {len(users)} users')

    def download_tweets(user):
        """Internal function to download all tweets of a user"""
        all_tweets = list()
        # get first 200 tweets
        tweets = api.user_timeline(screen_name=user,
                                   # 200 is the maximum allowed count
                                   count=200,
                                   include_rts=False,
                                   # Necessary to keep full_text
                                   # otherwise only the first 140 words are extracted
                                   tweet_mode='extended'
                                   )
        all_tweets.extend(tweets)

        if len(all_tweets) == 0:
            # return empty DataFrame for users without tweets
            return pd.DataFrame([])

        oldest_id = tweets[-1].id

        # get all following tweets
        while True:
            tweets = api.user_timeline(screen_name=user,
                                       # 200 is the maximum allowed count
                                       count=200,
                                       include_rts=False,
                                       max_id=oldest_id - 1,
                                       # Necessary to keep full_text
                                       # otherwise only the first 140 words are extracted
                                       tweet_mode='extended'
                                       )
            if len(tweets) == 0:
                break
            oldest_id = tweets[-1].id
            all_tweets.extend(tweets)

        print(f'Number of tweets downloaded from user {user}: {len(all_tweets)}')

        # save tweets per user in a DataFrame
        user_tweets = pd.DataFrame([[
            tweet.id_str,
            tweet.created_at,
            tweet.favorite_count,
            tweet.retweet_count,
            tweet.source,
            tweet.truncated,
            tweet.in_reply_to_status_id_str,
            tweet.lang,
            tweet.entities['hashtags'],
            tweet.full_text.encode("utf-8").decode("utf-8")]
            for tweet in all_tweets],
            columns=["id", "created_at", "favorite_count", "retweet_count", "source", "truncated",
                     "reply_to", "language", "hashtags", "text"])
        user_tweets['user'] = user

        return user_tweets

    for user in users:
        try:
            user_tweets = download_tweets(user)

        except error.RateLimitError:
            # Tweepy has a rate limit for request. If rate limit is reached, the processes is sleeping for 15 minutes
            # save tweets
            if save_path:
                tweets_df.to_csv(save_path + '.csv')
                tweets_df.to_pickle(save_path + '.pkl')

            print(f'Reached rate limit from tweepy after {len(tweets_df.user.unique())} users. Paused for 15 minutes\n')
            time.sleep(1000)
            # repeat request
            user_tweets = download_tweets(user)

        except error.TweepError:
            # User does not exist
            user_tweets = pd.DataFrame([])
            pass

        # Add user_tweets to overall DataFrame
        tweets_df = tweets_df.append(user_tweets, ignore_index=True)

    print(f'Downloaded tweets from {len(tweets_df.user.unique())} users')

    # save tweets
    if save_path:
        tweets_df.to_csv(save_path + '.csv')
        tweets_df.to_pickle(save_path + '.pkl')

    return tweets_df


if __name__ == '__main__':
    # connect to twitter
    api = get_twitter_api()

    # get twitter names
    users_df = pd.read_excel('data/congress_twitter_accounts.xlsx')
    print(f'Read {len(users_df)} politicians')
    # drop politicians without twitter accounts
    users_df = users_df.dropna(subset=['user'])

    print(f'Found {len(users_df)} twitter accounts')

    # load and save tweets
    tweets_df = get_user_tweets(api, users_df['user'], save_path='data/tweets_raw')


