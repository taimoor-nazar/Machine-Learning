#Detecting Spam email using label Spreading(Semi Supervised Learning)

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

#converting to lowercase

df['message'] = df['message'].str.lower()

#removing email-addresses URls and numbers

df['message'] = df['message'].apply(lambda x: re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', x))
df['message'] = df['message'].apply(lambda x: re.sub(r'http\S+', '', x))
df['message'] = df['message'].apply(lambda x: re.sub(r'\d+', '', x))

#removing special characters and punctuation

df['message'] = df['message'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

#removing stopwords

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['message'] = df['message'].apply(lambda x: re.sub(r'\b(?:' + '|'.join(stopwords.words('english')) + r')\b', '', x))

#Coverting to numerical format using tfidf vectorizer

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['message'])
y = df['label']

# Seperating which index has ham and which has spam
ham_indices = df[df['label'] == 'ham'].index
spam_indices = df[df['label'] == 'spam'].index

# Randomly select 20% from each for labelled data and 80% for unlabelled
ham_labelled, ham_unlabelled = train_test_split(ham_indices, test_size=0.8)
spam_labelled, spam_unlabelled = train_test_split(spam_indices, test_size=0.8)

# Combine labelled and unlabelled indices
labelled_indices = np.concatenate([ham_labelled, spam_labelled])
unlabelled_indices = np.concatenate([ham_unlabelled, spam_unlabelled])

# Setting unlabelled data as -1
y_final = np.full(df.shape[0], -1)
le = LabelEncoder()
y_encoded = le.fit_transform(df['label'])
y_final[labelled_indices] = y_encoded[labelled_indices]

#training the model

model = LabelSpreading()
model.fit(x, y_final)

#getting the accuracy
predictions = model.predict(x[unlabelled_indices])
predictions = le.inverse_transform(predictions)

error_report = classification_report(y[unlabelled_indices], predictions)

print(error_report)