import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("Placement_BeginnerTask01.csv")

df.drop(columns=["StudentID"], inplace=True)

df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map({
    "Yes": 1,
    "No": 0
})

df["PlacementTraining"] = df["PlacementTraining"].map({
    "Yes": 1,
    "No": 0
})

df["PlacementStatus"] = df["PlacementStatus"].map({
    "Placed": 1,
    "NotPlaced": 0
})

X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))