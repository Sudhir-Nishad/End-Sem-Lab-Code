import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "EC100": [85, 78, 92, 74, 88, 67, 94, 76, 84, 69],
    "IT101": [89, 74, 91, 73, 85, 65, 95, 75, 83, 70],
    "MA101": [90, 77, 93, 72, 87, 66, 96, 74, 82, 68],
    "PH100": [88, 75, 92, 70, 86, 64, 95, 73, 81, 67],
    "Internship_Eligibility": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)

X = df[["EC100", "IT101", "MA101"]]
y = df["Internship_Eligibility"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

naive_bayes = GaussianNB()

naive_bayes.fit(X_train, y_train)

y_pred = naive_bayes.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

X["Predicted_PH100"] = naive_bayes.predict(X)

print("\nPredicted Internship Eligibility:\n", X)