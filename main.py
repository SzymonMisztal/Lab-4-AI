import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


data = pandas.read_csv("glass.data", header=None)
data.columns = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

print(data)

X = data.drop(['ID', 'Type'], axis=1)
y = data['Type']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)



print('////////// Vanilla //////////\n')



print('/// Naive Bayes ///\n')
grid_search = GridSearchCV(GaussianNB(), {}, cv=5)

grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_val)
print(classification_report(y_val, y_pred, zero_division=1))

print('/// Decision Tree ///\n')
grid_search = GridSearchCV(DecisionTreeClassifier(), {'max_depth': [None, 5, 10]}, cv=5)

grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_val)
print(classification_report(y_val, y_pred, zero_division=1))



print('////////// Normalization //////////\n')
processingN = MinMaxScaler()

processingN.fit(X_train)
X_train_processed = processingN.transform(X_train)
X_val_processed = processingN.transform(X_val)



print('/// Naive Bayes ///\n')
grid_search = GridSearchCV(GaussianNB(), {}, cv=5)

grid_search.fit(X_train_processed, y_train)

y_pred = grid_search.predict(X_val_processed)
print(classification_report(y_val, y_pred, zero_division=1))

print('/// Decision Tree ///\n')
grid_search = GridSearchCV(DecisionTreeClassifier(), {'max_depth': [None, 5, 10], 'criterion': ['entropy', 'gini']}, cv=5)

grid_search.fit(X_train_processed, y_train)

y_pred = grid_search.predict(X_val_processed)
print(classification_report(y_val, y_pred, zero_division=1))



print('////////// Standardization //////////\n')
processingS = StandardScaler()

processingS.fit(X_train)
X_train_processed = processingS.transform(X_train)
X_val_processed = processingS.transform(X_val)



print('/// Naive Bayes ///\n')
grid_search = GridSearchCV(GaussianNB(), {}, cv=5)

grid_search.fit(X_train_processed, y_train)

y_pred = grid_search.predict(X_val_processed)
print(classification_report(y_val, y_pred, zero_division=1))

print('/// Decision Tree ///\n')
grid_search = GridSearchCV(DecisionTreeClassifier(), {'max_depth': [None, 5, 10], 'criterion': ['entropy', 'gini']}, cv=5)

grid_search.fit(X_train_processed, y_train)

y_pred = grid_search.predict(X_val_processed)
print(classification_report(y_val, y_pred, zero_division=1))


