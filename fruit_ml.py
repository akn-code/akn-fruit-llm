# 1. Import libraries
import pandas as pd # <-- Import pandas
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 2. Load Your Own Data
try:
    data = pd.read_csv("my_fruit.csv")
except FileNotFoundError:
    print("Error: 'my_fruit.csv' not found. Make sure it's in the same folder.")
    exit()

# 3. Prepare the Data
# The model needs numbers, not text like "bumpy" or "orange".
# We'll map them to numbers.
data['texture'] = data['texture'].map({'bumpy': 0, 'smooth': 1})
data['fruit_name'] = data['fruit_name'].map({'orange': 0, 'apple': 1})

# Separate features (X) from the target (y)
features = ['weight', 'texture']
target = 'fruit_name'

X = data[features]
y = data[target]

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create and Train the Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("Model trained on 'my_fruit.csv'!")

# 6. Test the Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy on test data: {accuracy * 100:.2f}%")

new_data = pd.DataFrame(
    [[160, 0]],                               # The values
    columns=['weight', 'texture']             # <-- The critical part: names must match!
) 

# 3. Predict using the DataFrame
prediction = model.predict(new_data)

# Map the number back to a name
fruit_names = {0: 'orange', 1: 'apple'}
print(f"Prediction for {new_data} is: {fruit_names[prediction[0]]}")
