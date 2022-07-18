import pandas as pd
from keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding


df = pd.read_csv("Sentiment_Sample.csv")

#print(df.head(10))

review_df = df[['Answer','New_Sentiment']]

#print(review_df.shape)

# Dropping out Neutral sentiments for the sake of binary classification

# review_df = review_df[review_df['New_Sentiment']!='neu']
#
# print(review_df.shape)

# Checking the value count of positive and negative sentiments

distinct_count = review_df["New_Sentiment"].value_counts()
print(distinct_count)

#converting categorical values to numeric
# 0 = positive, 1= negative
sent_label = review_df.New_Sentiment.factorize()
print(sent_label)

#retrieving all the text from review_df

sentiment = review_df.Answer.values

#tokenizing words

tokenizer = Tokenizer(num_words = 500)
tokenizer.fit_on_texts(sentiment)
vocab_size = len(tokenizer.word_index)+1
#Replacing the words with their assigned numbers

encoded = tokenizer.texts_to_sequences(sentiment)

#Padding the sentences to have equal length

padded_sequence = pad_sequences(encoded, maxlen = 300)

#Building Text Classifier

embedding_vec_len = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vec_len, input_length = 300))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout = 0.5, recurrent_dropout = 0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# model = Sequential()
# model.add(Dense(100, activation = 'relu', input_length = 300))
# model.add(Dense(100, activation = 'relu'))
# model.add(Dense(100, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation = 'softmax'))
# model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# print(model.summary())

#Training

hist = model.fit(padded_sequence,sent_label[0],validation_split = 0.2, epochs = 5, batch_size = 40)

#Predicting output

def predict_sentiments(Comment):
    tw = tokenizer.texts_to_sequences([Comment])
    tw = pad_sequences(tw,maxlen = 300)
    pred = int(model.predict(tw).round().item())
    print("Predicted Output: ", sent_label[1] [pred])

test_1 = "I loved working here."
predict_sentiments(test_1)

test_2 = "Working here was the worst experience of my life"
predict_sentiments(test_2)