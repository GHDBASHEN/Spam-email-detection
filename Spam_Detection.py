import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('spam.csv', encoding='utf-8-sig')
#print(data.columns)
data = data.iloc[:, [0, 1]]  # Selecting only the first two columns if they contain label and message
data.columns = ['label', 'message']  # Rename them for convenience
#data = data[['v1', 'v2']]  # Selecting relevant columns
#data.columns = ['label', 'message']  # Renaming columns
print(data.head())

# Map labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text to vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
