import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("teleCust.csv")

cust = df['custcat'].value_counts()
print(cust)

#Plot a histogram based on income
hist_cust = df.hist(column = 'income', bins = 50)
plt.show()

print(df.columns)

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside']].values
print(X[0:5])

y = df[['custcat']].values
print(y[0:5])

#Normalise the data
X = StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print("Train set", X_train.shape, y_train.shape)
print("Test set", X_test.shape, y_test.shape)

#KNeighbors Classifier
k = 6
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
print(knn)

pred = knn.predict(X_test)
pred_train = knn.predict(X_train)
print(pred[0:5])

#Calculate the accuracy

print("Train set accuracy", accuracy_score(pred_train, y_train))
print("Test set accuracy", accuracy_score(y_test, pred))

#Calculating accuracies for different ks.

ks = 10
mean_acc = np.zeros((ks-1))
std_acc = np.zeros((ks-1))
CM = []

for n in range(1, ks):
    knn_1 = KNeighborsClassifier(n_neighbors=n)
    knn_1.fit(X_train, y_train)
    pred_ks = knn_1.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, pred_ks)

    std_acc[n-1] = np.std(pred_ks == y_test)/np.sqrt(pred_ks.shape[0])

print(mean_acc)

plt.plot(range(1, ks), mean_acc, 'g')
plt.fill_between(range(1, ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()


