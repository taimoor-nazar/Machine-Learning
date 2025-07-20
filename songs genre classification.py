#Classifying song genres using Random Forest Classifier on Spotify dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from imblearn.over_sampling import SMOTE


df = pd.read_csv('SpotifyFeatures.csv')

#exploring the dataset
print(df.describe()) 
print(df.info())

#Exploratory Data Analysis of Numerical Features

num_cols = []
for col in df.columns:

  if df[col].dtype == 'int64' or df[col].dtype == 'float64':
    num_cols.append(col)
'''
    plt.show()
    sns.boxplot(data=df, x=col, y='genre')
    plt.show()
    sns.histplot(x=df[col])
    plt.show()

sns.heatmap(df[num_cols].corr(), annot=True, cmap = 'coolwarm')
plt.show()
'''
#Some Features have extremly high correlation
'''
sns.scatterplot(data =df, x='energy', y='loudness')
plt.show()

top_genres = df['genre'].value_counts().head(6).index
df_subset = df[df['genre'].isin(top_genres)]
sns.pairplot(df_subset[['popularity','energy', 'valence', 'tempo', 'danceability', 'genre']], hue='genre', corner=True, plot_kws={'alpha': 0.5})
plt.show()
'''
#Conclusion of EDA: We can idenitfy some realtionships and correlations between some feature for e.g:
#                   danceability and energy have high correlation and many of those songs belongs to the electronic genre
#                   pop  music is highly popular

# Now we check the non numeric columns(mode and key)

'''
le_mode, le_key,  = LabelEncoder()

df['mode'] = le_mode.fit_transform(df['mode'])
df['key'] = le_key.fit_transform(df['key'])

sns.countplot(data=df, x='mode', hue='genre')
plt.show()

sns.countplot(data=df, x='key', hue='genre')
plt.show()

print(df.groupby('genre')['mode'].mean())
print(df.groupby('genre')['key'].mean())
'''

#Conclusion : There is very little variation bw genres for different modes and keys hence they are not useful features to determine genres

#Preprocessing and training a classification model:

#Selecting only useful features

df['genre'] = df['genre'].replace("Children’s Music", "Children's Music") #removing duplicate genres
num_cols.remove('duration_ms') #duration is removed as it is not useful for genre classification


#After initial training I combined similar song genres with low accuracy to reduce no of genres and improve accuracy

genre_map = {
    'Hip-Hop': 'Urban',
    'Rap': 'Urban',
    'R&B': 'Urban',

    'Rock': 'AltRock',
    'Alternative': 'AltRock',
    'Indie': 'AltRock',

    'Folk': 'FolkWorld',
    'World': 'FolkWorld',
    'Country': 'FolkWorld',

    'Pop': 'PopGeneral',
    'Dance': 'PopGeneral',
    'Soul': 'PopGeneral',

    'Jazz': 'JazzBlues',
    'Blues': 'JazzBlues',

    "Children's Music": 'ChildMusic',

    'Reggae': 'ReggaeStyle',
    'Ska': 'ReggaeStyle',

    # Keeping genres with high accuracy unchanged
    'A Capella': 'A Capella',
    'Anime': 'Anime',
    'Classical': 'Classical',
    'Comedy': 'Comedy',
    'Electronic': 'Electronic',
    'Movie': 'Movie',
    'Opera': 'Opera',
    'Reggaeton': 'Reggaeton',
    'Soundtrack': 'Soundtrack',
}

# Apply mapping
df['genre'] = df['genre'].map(genre_map)
le_genre = LabelEncoder()
x = df[num_cols]
y = le_genre.fit_transform(df['genre'])

model = RandomForestClassifier(class_weight = 'balanced')

#Applying SMOTE to handle class imbalance 
sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(x, y)

#Splitting Data
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.2, random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred = le_genre.inverse_transform(y_pred)
y_test = le_genre.inverse_transform(y_test)

#Evaluation
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred) )

#Confusion Matrix
genre_labels = le_genre.classes_
cm = confusion_matrix(y_test, y_pred, labels=genre_labels)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized * 100, annot=True, fmt=".1f", cmap='YlGnBu',xticklabels=genre_labels, yticklabels=genre_labels)

plt.xlabel('Predicted Genre')
plt.ylabel('Actual Genre')
plt.title('Confusion Matrix (% per Actual Genre)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()