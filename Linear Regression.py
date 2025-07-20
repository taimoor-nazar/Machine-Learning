import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#importing dataset and converting it into a datafram
data = fetch_california_housing(as_frame=True)
df = data.frame

#selecting my features in x and my target in y
x = df[['HouseAge', 'AveRooms', 'Latitude', 'Longitude']]
y = df['MedHouseVal']

#splitting the dataset for training and testing(80% training and 20%testing)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42) #random state is used to prevent random data split

model = LinearRegression() #initialize the model
model.fit(x_train, y_train) #training the model

prediction = model.predict(x_test) #make our predictions for the test data

r2 = r2_score(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
#R^2 score and mean squared error are used to check the model's accuracy by comparing the actual target values and the predicted values

print("R^2 Score: ", r2)
print("Mean Squared Error: ", mse)

#used to show the difference bw actual and predicted values
#red line shows the ideal scenario and blue shows the predictions

plt.figure(figsize=(8, 6))
plt.scatter(y_test, prediction, alpha=0.5, color='blue')
plt.xlabel("Actual Median House Value($100000)")
plt.ylabel("Predicted Median House Value($100000)")
plt.title("Actual vs Predicted House Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.show()


