import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('drug200.csv')

print(df.columns)
print(df.shape)

X= df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

#Convert categorical variable into dummy/indicator variables.
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['HIGH','LOW','NORMAL'])
X[:,2] = le_BP.transform(X[:,2])

le_ch = preprocessing.LabelEncoder()
le_ch.fit(['NORMAL','HIGH'])
X[:,3] = le_ch.transform(X[:,3])

y = df[['Drug']].values
print(y[0:5])

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 14)
print("The shape of the training set is: ",X_train.shape,y_train.shape)
print("The shape of the testing set is: ", X_test.shape,y_test.shape)


dt = DecisionTreeClassifier(criterion='entropy', max_depth = 4)

dt.fit(X_train,y_train)
pred = dt.predict(X_test)
print(pred[0:5])
print(y_test[0:5])

print("The accuracy of the Decision Tree is: ",accuracy_score(pred,y_test))