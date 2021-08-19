from keras import Sequential
from keras.layers import Dense
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

df = pd.read_csv("ChurnData.csv")
print(df.shape)
print(df.isnull().sum())
print(df.columns)

#TTS
X = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',
       'callcard', 'wireless', 'longmon', 'tollmon', 'equipmon', 'cardmon',
       'wiremon', 'longten', 'tollten', 'cardten', 'voice', 'pager',
       'internet', 'callwait', 'confer', 'ebill', 'loglong', 'logtoll',
       'lninc', 'custcat']].values

y = df[['churn']].values

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state = 4)


#Splitting data

df_cols = df.columns
predictors = df[df_cols[df_cols != 'churn']]
target = df['churn']

#Sanity check
print(predictors.head())
print(target.head())

pred_norm = (predictors-predictors.mean())/predictors.std()
print(pred_norm.head())

n_cols = pred_norm.shape[1]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Model

def ChurnModel():
    model = Sequential()
    model.add(Dense(5,activation = 'relu', input_shape=(n_cols,)))
    model.add(Dense(5,activation = 'relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

model = ChurnModel()

model.fit(X_train,y_train,validation_data =(X_test,y_test),epochs = 100, verbose = 2)
score = model.evaluate(X_test,y_test,verbose= 0)
print('Accuracy: {}% \n Error: {}'.format(score[1], 1 - score[1]))