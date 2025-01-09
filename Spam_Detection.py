import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset with correct encoding to handle BOM
try:
    data = pd.read_csv('spam.csv', encoding='utf-8-sig')
except FileNotFoundError:
    print("Error: Dataset file 'spam.csv' not found.")
    exit()

# Display the column names for debugging
print("Columns in dataset:", data.columns.tolist())

# Display the entire dataset
print("Full dataset:\n", data)

# Select and rename the first two relevant columns
if len(data.columns) >= 2:
    data = data.iloc[:, :2]  # Selecting only the first two columns
    data.columns = ['label', 'message']
else:
    print("Error: Dataset does not contain enough columns.")
    exit()

# Display the first few rows for confirmation
print(data.head())

# Map labels to binary values (ham: 0, spam: 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data into training and testing sets
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text to vectors using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
