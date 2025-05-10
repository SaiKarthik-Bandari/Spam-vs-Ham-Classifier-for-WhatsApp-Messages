import pandas as pd
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv', sep='\t', header=None, names=['label', 'message'])
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

stop_words = set(stopwords.words('english'))

vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy*100:.2f}%")

joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
