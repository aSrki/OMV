from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

n_frames = 7

train_df = pd.read_csv(f"training/area/view_a/training_data_a_{n_frames}.csv")
test_df = pd.read_csv(f"testing/area/view_a/testing_data_a_{n_frames}.csv")

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(
    n_neighbors=150, 
    metric='minkowski', 
    p=2, 
    weights='distance'
)

print("Treniranje kNN modela...")

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print("\n--- REZULTATI (kNN) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nIzveštaj:")
print(classification_report(y_test, y_pred))