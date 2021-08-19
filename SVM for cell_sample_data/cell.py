import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import jaccard_similarity_score, accuracy_score

df = pd.read_csv('cell_samples.csv')
print(df.columns)
print(df.shape)
print(df.dtypes)

df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc'] = df['BareNuc'].astype('int')
print(df.dtypes)

X = df[['ID', 'Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']].values
print(X[0:5])

y = df[['Class']].values
print(y[0:5])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 14)
print("The training set is:", X_train.shape,y_train.shape)
print("The testing set is:",X_test.shape,y_test.shape)

clf = svm.SVC(kernel= 'rbf')
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

#calculate score using jaccard
score = jaccard_similarity_score(y_test,pred)
print(score)

#calculate score using accuracy_score
score_1 = accuracy_score(y_test,pred)
print(score_1)

