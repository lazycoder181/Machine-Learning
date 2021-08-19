import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, jaccard_similarity_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('wine.csv')
print(df.columns)
print(df.shape)

X = df[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
       'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
       'proanthocyanins', 'color_intensity', 'hue', 'od280', 'proline']].values
print(X[0:5])

y = df[['class_label']].values
print(y[0:5])

#Preprocessing

X = StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

#Train-Test-Split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state = 4)
print("The training set:", X_train.shape, y_train.shape)
print("The testing set:", X_test.shape,y_test.shape)

#KNN

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(pred[0:5])
print(y_test[0:5])
print("The accuracy score of knn is:", accuracy_score(y_test,pred))

#Decision Tree

dt = DecisionTreeClassifier(criterion='entropy', max_depth = 4)
dt.fit(X_train,y_train)
pred_dt = dt.predict(X_test)
print(y_test[0:5])
print(pred_dt[0:5])
print("The accuracy score of the decision tree is:", accuracy_score(y_test,pred_dt))

#Logistic Regression

logreg = LogisticRegression(C= 0.01, solver='liblinear')
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(y_test[0:5])
print(pred_logreg[0:5])

pred_prob = logreg.predict_proba(X_test)
print(pred_prob)

eval = log_loss(y_test,pred_prob)
print("The log loss of logreg is:",eval)

#SVM
clf = svm.SVC(kernel= 'rbf')
clf.fit(X_train,y_train)
pred_clf = clf.predict(X_test)

score = jaccard_similarity_score(y_test,pred_clf)
print("The accuracy score of SVM using jaccard is:",score)


