import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Union

class FruitClassifier:
    """
    A class to classify fruits based on their features (weight and texture) using a Decision Tree.
    
    Think of this class as a robot that learns to tell apples from oranges.
    """

    def __init__(self):
        """
        Initialize the classifier.
        This sets up the empty 'brain' (model) and the 'translators' (encoders) 
        that turn words into numbers.
        """
        # The Decision Tree is our machine learning model.
        # It asks a series of yes/no questions to make a decision.
        self.model = DecisionTreeClassifier()
        
        # LabelEncoders are like translators. Computers understand numbers, not words.
        # These will turn 'bumpy' into 0 and 'smooth' into 1, etc.
        self.le_texture = LabelEncoder()
        self.le_fruit = LabelEncoder()
        
        # These are the columns we will use to learn.
        self.features = ['weight', 'texture']
        # This is what we want to predict.
        self.target = 'fruit_name'

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Loads data from a CSV file into a pandas DataFrame.
        A DataFrame is like a programmable Excel spreadsheet.
        """
        try:
            data = pd.read_csv(filepath)
            return data
        except FileNotFoundError:
            print(f"Error: '{filepath}' not found. Please check the file name and location.")
            return pd.DataFrame()

    def preprocess(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Preprocesses the data by turning text into numbers.
        
        Args:
            data: The data to process.
            fit: If True, the translators (encoders) learn the mapping (e.g., bumpy=0).
                 If False, they just use what they already learned.
                 We use fit=True for training data, and fit=False for new data we want to predict.
        """
        # We work on a copy so we don't mess up the original data
        data = data.copy()
        
        if fit:
            # fit_transform: Learn the words AND convert them to numbers
            data['texture'] = self.le_texture.fit_transform(data['texture'])
            if self.target in data.columns:
                data[self.target] = self.le_fruit.fit_transform(data[self.target])
        else:
            # transform: Just convert using what we learned before
            try:
                data['texture'] = self.le_texture.transform(data['texture'])
                if self.target in data.columns:
                    data[self.target] = self.le_fruit.transform(data[self.target])
            except ValueError as e:
                # This happens if we see a word we've never seen before (e.g., 'fuzzy')
                print(f"Error during encoding: {e}")
                
        return data

    def train(self, data: pd.DataFrame):
        """
        Trains the Decision Tree model.
        This is where the 'learning' happens.
        """
        if data.empty:
            print("No data to train on.")
            return

        # 1. Prepare the data (turn words to numbers)
        processed_data = self.preprocess(data, fit=True)
        
        # 2. Separate the questions (X) from the answers (y)
        X = processed_data[self.features]
        y = processed_data[self.target]

        # 3. Split data into study material (train) and a final exam (test)
        # test_size=0.2 means 20% of data is saved for the exam.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Teach the model using the study material
        self.model.fit(X_train, y_train)
        print("Model trained successfully. The robot has learned!")
        
        # 5. Give the robot the final exam to see how well it did
        self.evaluate(X_test, y_test)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluates the model and prints a report card (metrics).
        """
        # Ask the model to predict the answers for the exam questions
        predictions = self.model.predict(X_test)
        
        # Compare predictions to the real answers
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n--- Evaluation Results ---")
        print(f"Accuracy: {accuracy * 100:.2f}% (Percentage of correct answers)")
        
        print("\nClassification Report (Detailed breakdown):")
        # We pass 'labels' to ensure all fruits are listed, even if some weren't in the exam
        print(classification_report(y_test, predictions, labels=range(len(self.le_fruit.classes_)), target_names=self.le_fruit.classes_))
        
        print("\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
        print(confusion_matrix(y_test, predictions))

    def predict(self, weight: int, texture: str) -> str:
        """
        Predicts the fruit name for a new fruit.
        
        Args:
            weight: The weight of the fruit in grams.
            texture: The texture ('smooth' or 'bumpy').
        """
        # Create a tiny dataframe for this one fruit
        input_data = pd.DataFrame([[weight, texture]], columns=self.features)
        
        # Convert words to numbers using the translators we already trained
        try:
            processed_input = self.preprocess(input_data, fit=False)
        except Exception:
             return "Error: Could not process input (maybe unknown texture?)"

        # Ask the model to predict (it returns a number)
        prediction_idx = self.model.predict(processed_input[self.features])[0]
        
        # Convert the number back to a word (e.g., 0 -> 'orange')
        return self.le_fruit.inverse_transform([prediction_idx])[0]

    def visualize_tree(self, filename: str = 'fruit_tree.png'):
        """
        Draws a picture of the decision tree logic and saves it.
        """
        plt.figure(figsize=(10, 6))
        plot_tree(self.model, feature_names=self.features, class_names=self.le_fruit.classes_, filled=True)
        plt.savefig(filename)
        print(f"\nDecision tree visualization saved to '{filename}'. Check this file to see how the robot thinks!")

if __name__ == "__main__":
    # Create our classifier instance
    classifier = FruitClassifier()
    
    # 1. Load the data
    print("Loading data...")
    df = classifier.load_data("my_fruit.csv")
    
    if not df.empty:
        # 2. Train the model
        print("Training model...")
        classifier.train(df)
        
        # 3. Visualize the brain
        classifier.visualize_tree()

        # 4. Make a prediction for a new fruit
        test_weight = 160
        test_texture = 'bumpy'
        print(f"\n--- Testing a new fruit ---")
        print(f"Fruit: {test_weight}g, {test_texture}")
        result = classifier.predict(test_weight, test_texture)
        print(f"Prediction: {result}")
