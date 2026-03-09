from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

n_frames = 5

train_df = pd.read_csv(f"training/area/view_a/training_data_a_{n_frames}.csv")
test_df = pd.read_csv(f"testing/area/view_a/testing_data_a_{n_frames}.csv")

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

rf = RandomForestClassifier(
    n_estimators=10, 
    max_depth=None, 
    min_samples_split=2,
    criterion='gini',
    random_state=8346,
    n_jobs=-1
)

print("Treniranje Random Forest modela...")
rf.fit(X_train, y_train) 

y_pred = rf.predict(X_test)

print("\n--- REZULTATI (Random Forest) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nIzveštaj:")
print(classification_report(y_test, y_pred))