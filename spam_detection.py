import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("spam.csv")

# Convert labels
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})

# Features and Labels
X = data['v2']
y = data['v1']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
predictions = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Test Message
message = ["Congratulations! You won a free gift."]

message_vec = vectorizer.transform(message)
result = model.predict(message_vec)

if result[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")
