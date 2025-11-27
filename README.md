# akn-fruit-llm

## fruit_ml.py

A machine learning script that classifies fruits (oranges vs apples) based on their weight and texture using a Decision Tree classifier.

### What it does:

1. **Loads data** from `my_fruit.csv` containing fruit characteristics
2. **Preprocesses the data** by converting categorical values to numerical:
   - Texture: 'bumpy' → 0, 'smooth' → 1
   - Fruit name: 'orange' → 0, 'apple' → 1
3. **Trains a Decision Tree model** using weight and texture as features
4. **Evaluates accuracy** on test data (80/20 train-test split)
5. **Makes predictions** on new fruit samples

### Features used:
- `weight`: The weight of the fruit
- `texture`: The surface texture (bumpy or smooth)

### Output:
- Training confirmation message
- Model accuracy percentage on test data
- Sample prediction for a fruit with weight=160 and bumpy texture
