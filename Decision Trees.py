import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#dataset downloaded from github
df = pd.read_csv('titanic.csv')

#remove unnecessary columns
inputs = df.drop(['Survived', 'PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns')
#set target
target = df['Survived']

#We need to convert Male and Female inputs to 1s or 0s
le_sex = LabelEncoder()
inputs['Sex_n'] = le_sex.fit_transform(inputs['Sex'])
inputs_n = inputs.drop('Sex', axis='columns')

#Fill missing age values with the median age
inputs_n['Age'] = inputs_n['Age'].fillna(inputs_n['Age'].median())

#Initialize the model
model = tree.DecisionTreeClassifier()

#Split the data
x_train , x_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.2)

#Train the model
model.fit(x_train, y_train)

#get predictions
prediction = model.predict(x_test)

#check accuracy score
accuracy = model.score(x_test, y_test)
print("Score: ", accuracy)

#Initilize confusion matrix
cm = confusion_matrix(y_test, prediction)

#Plot the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
