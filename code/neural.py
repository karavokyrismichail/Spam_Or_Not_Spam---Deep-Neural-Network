import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import keras
from keras.models import Sequential
from keras.layers import Dense


sns.set()

df = pd.read_csv (r'spam_or_not_spam.csv')
df = df.dropna(subset=['email'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['email'])
X = pd.DataFrame(X.todense())
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X_train)
y = df['label']
#split the dataframe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

model = Sequential()
model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x = X_train, y = y_train, batch_size = 5, epochs=8)
y_pred = (model.predict(X_test) > 0.5).astype("int32")
prec_score = metrics.precision_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)
rec_score = metrics.recall_score(y_test, y_pred)
print(prec_score)
print(f1_score)
print(rec_score)