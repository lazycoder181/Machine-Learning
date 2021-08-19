import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential

df = pd.read_csv('banknotes.csv')
print(df.columns)
print(df.shape)
print(df.isnull().sum())

X = df[['variance', 'skewness', 'curtosis', 'entropy']].values
y = df[['result']].values

X = StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])
print(y[0:5])

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)

df_cols = df.columns
predictors = df[df_cols[df_cols != 'result']]
target = df['result']

pred_norm = (predictors - predictors.mean())/predictors.std()
print(pred_norm.head())

n_cols = pred_norm.shape[1]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Model defining

def Originality():
    model = Sequential()
    model.add(Dense(100, activation = 'relu', input_shape=(n_cols,)))
    model.add(Dropout(0.2))
    model.add(Dense(75, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

model = Originality()

training = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs = 25)
pred = model.predict(X_test)
print(pred)

score = model.evaluate(X_test,y_test)
print("Accuracy: {}% \n Error: {}".format (score[1], 1-score[1]))

plt.plot(training.history['val_loss'],'r')
plt.xlabel("Epochs")
plt.ylabel('Validation score')
plt.show()