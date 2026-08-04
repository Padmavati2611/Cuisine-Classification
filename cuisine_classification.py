from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
# Load dataset
df = pd.read_csv("Dataset .csv")

# Display first 5 rows
print(df.head())
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Select input features
features = [
    "City",
    "Average Cost for two",
    "Price range",
    "Aggregate rating",
    "Votes",
    "Has Table booking",
    "Has Online delivery"
]

# Target column
target = "Cuisines"

print(features)
print(target)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# Encode input categorical columns
df["City"] = encoder.fit_transform(df["City"])

df["Has Table booking"] = encoder.fit_transform(
    df["Has Table booking"]
)

df["Has Online delivery"] = encoder.fit_transform(
    df["Has Online delivery"]
)

# Encode target cuisine
df["Cuisines"] = encoder.fit_transform(
    df["Cuisines"]
)

print(df.head())
# Input data
X = df[features]

# Output label
y = df[target]

print("Input Features:")
print(X.head())

print("\nTarget:")
print(y.head())
from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

# Train model
model.fit(
    X_train,
    y_train
)

print("Model training completed!")
# Predict cuisine on test data
y_pred = model.predict(X_test)

print(y_pred[:10])
model.fit(X_train, y_train)
# Predict on test data
y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred[:10])
# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# Precision
precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

print("Precision:", precision)


# Recall
recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

print("Recall:", recall)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

print(classification_report(y_test, y_pred))
# Keep only cuisines with more than 100 restaurants

cuisine_counts = df["Cuisines"].value_counts()

popular_cuisines = cuisine_counts[
    cuisine_counts > 100
].index

df = df[df["Cuisines"].isin(popular_cuisines)]
model.fit(X_train, y_train)
print("Number of unique cuisines:")
print(df["Cuisines"].nunique())

print("\nTop cuisines:")
print(df["Cuisines"].value_counts().head(10))
# Combine useful text information

df["combined_features"] = (
    df["Restaurant Name"].astype(str)
    + " "
    + df["City"].astype(str)
)
tfidf = TfidfVectorizer(
    max_features=500
)

X_text = tfidf.fit_transform(
    df["combined_features"]
)
y = df["Cuisines"]
encoder = LabelEncoder()

y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test,y_pred))

print(
classification_report(y_test,y_pred)
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))

plt.imshow(cm)

plt.title("Cuisine Classification Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.colorbar()

plt.savefig("confusion_matrix.png")
plt.show()