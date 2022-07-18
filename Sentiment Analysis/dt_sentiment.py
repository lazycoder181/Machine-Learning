import pandas as pd
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("sample_revised.csv")
print(df.columns)
print(df.shape)

#Dropping unnecessary columns
df_1 = df.drop(['Unnamed: 0','element_id','CommentSent', 'Question','ProductGroupEntity', 'Answer', 'X.y', 'QuestionName.y', 'Division','answer_split', 'isPosOrNeg', 'Sentiment' ], axis = 1)

print(df_1.columns)

#retrieving all the text from review_df

comments = df_1.Comment.values
#tokenizing words

tokenizer = Tokenizer(num_words = 500)
tokenizer.fit_on_texts(comments)
vocab_size = len(tokenizer.word_index)+1

#Replacing the words with their assigned numbers

encoded = tokenizer.texts_to_sequences(comments)

#Padding the sentences to have equal length

X = pad_sequences(encoded, maxlen = 300)

# X = df_1[['Comment']].values
# print(X[0:5])

#Converting categorical variable into dummy variables

y = df_1[['New_Sentiment']].values
print(y[0:5])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 14)
print("Shape of training set: ", X_train.shape,y_train.shape )
print("Shape of test set: ", X_test.shape, y_test.shape)

dt = DecisionTreeClassifier(criterion = 'entropy', max_depth= 4)

dt.fit(X_train,y_train)
pred = dt.predict(X_test)
print(pred)
print(y_test)

print("The accuracy of the Decision Tree is: ", accuracy_score(pred,y_test))
