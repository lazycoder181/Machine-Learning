import pandas as pd
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Create dataframe
df = pd.read_csv("Final_Sentiment_Analysis_1.csv")

#Initialization
analyser = SentimentIntensityAnalyzer()

#Creating new column for calculating polarity scores for pos,neg,neu and compound
df['Sentiment'] = df['Comment'].map(lambda x: analyser.polarity_scores(x))
print(df.head(10))


def calculate(data):
    if data['compound'] > 0.52:
        if data['neg'] > 0:
            data = 'neg'
        else:
            data = 'pos'

    elif data['compound'] < 0.48:
        data = 'neg'

    elif 0.47 < data['compound'] < 0.53:
        if data['neg'] > 0:
            data = 'neg'
        else:
            data = 'neu'
    return data

# def calculate(data):
#     if data['neg'] > data['neu'] and data['neg'] > data['pos']:
#         data = 'neg'
#     elif data['neu'] > data['neg'] and data['neu'] > data['pos']:
#         data = 'neu'
#     elif data['pos'] > data['neg'] and data['pos'] > data['neu']:
#         data = 'pos'
#     return data


df['New_Sentiment'] = df['Sentiment'].apply(calculate)
print(df)

df.to_csv('sample_revised.csv')
