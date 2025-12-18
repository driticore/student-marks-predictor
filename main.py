import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Loading data
data = pd.read_csv('score.csv', usecols=['Hours', 'Scores'])
df = pd.DataFrame(data)

# Creating 2D Array ( features & labels )
X = df[['Hours']]   # Independent Variable
y = df['Scores']     # Dependent Variable

# Splitting Data (Train vs Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Creating the Model ( More like declearing )
model = LinearRegression()

### 
# Training the Model
# - It learns the slope (m) which is how much marks increase /hr and,
# - The intercept (c) which is the base marks
###
model.fit(X_train, y_train)


# Predictions of marks for future study hours
y_pred = model.predict(X_test)

hours = [[5]]
predicted_marks = model.predict(hours)
print(f"Predicted Marks for 5 hours study: {predicted_marks[0]:.2f}")

# Visualizing the model
plt.scatter(X, y)
plt.plot(X, model.predict(X), linewidth=2)
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Student Marks Predictor")
plt.show()

### Model uses the stright line equation
# Basically: 
# - y = Marks
# - m = Mark Increased Per Hour
# - x = Hours Studied
# - c = Base Marks

# Marks = (9.8 * Hours) + 2.44
###
print("Slope:", model.coef_)
print("Intercept:", model.intercept_)
