from nltk import SnowballStemmer, word_tokenize, TweetTokenizer, casual_tokenize
import pandas as pd
from tqdm import tqdm

# From Lab 5 - code snippet
# background: generell method structure
# foreground: tested different processing steps, tested different lexicons, adjusted method with stem comparision and neutral words

# Lexicon from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
# class can handle different lexicons. "emotion_lexicon" is recommended and default because of lexicon size.
emotion_intensity_lexicon_path = 'data/NRC-Suite-of-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/' \
                                 'NRC-Emotion-Intensity-Lexicon-v1/NRC-Emotion-Intensity-Lexicon-v1.txt'

emotion_lexicon_path = 'data/NRC-Suite-of-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/' \
                       'NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'


def text_emotion(df, column, path_to_root='', lexicon='emotion_lexicon'):
    '''
    Takes a DataFrame and a specified column of text and adds 10 columns to the
    DataFrame for each of the 10 emotions in the NRC Emotion Lexicon, with each
    column containing the value of the text in that emotions
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns
    '''

    filepath = path_to_root
    if lexicon == 'emotion_lexicon':
        filepath += emotion_lexicon_path
    elif lexicon == 'emotion_intensity_lexicon':
        # note: emotion_lexicon contains around 3 times more words than emotion_intensity_lexicon
        filepath += emotion_intensity_lexicon_path
    else:
        filepath += lexicon

    # get emotion intensity scores from lexicon
    emolex_df = pd.read_csv(filepath,
                            names=["word", "emotion", "score"],
                            sep='\t', header=0)

    # create pivot table
    emolex_words = emolex_df.pivot(index='word',
                                   columns='emotion',
                                   values='score').reset_index()
    emolex_words.fillna(0, inplace=True)
    # adding neutral for words not labeled with emotions
    emolex_words['neutral'] = 0

    # drop word == 0, looks like an error in pivot() as 0 is not in emolex_df
    emolex_words = emolex_words[emolex_words.word != 0]

    # get list of emotions
    emotions = emolex_words.columns.drop('word')

    # create tweet-emotion-matrix
    emo_df = pd.DataFrame(0., index=df.index, columns=emotions)

    # copy column to prevent changes on original text
    df_column = df[column].copy()

    # tokenize tweets if str is provided
    if type(df[column].iloc[0]) == str:
        tokenizer = TweetTokenizer()
        df_column = df_column.apply(str.lower)
        df_column = df_column.apply(tokenizer.tokenize)

    stemmer = SnowballStemmer("english")
    emolex_words['stem'] = emolex_words.word.apply(stemmer.stem)

    for i, text in df_column.items():
        for word in text:
            stem = stemmer.stem(word)

            # get matches with stem
            emo_score = emolex_words[emolex_words.stem == stem]
            if not emo_score.empty:
                # in case of multiple matches with word stem
                if word in emo_score.word.values:
                    # match with exact word
                    emo_df.loc[i] = emo_df.loc[i] + emo_score[emo_score.word == word][emotions].values[0]
                else:
                    # mean of all stem matches
                    emo_df.loc[i] = emo_df.loc[i] + emo_score[emotions].mean().values
            else:
                emo_df.at[i, 'neutral'] += 1.

    return emo_df
