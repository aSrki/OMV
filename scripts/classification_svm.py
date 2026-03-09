from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

n_frames = 3

train_df_a = pd.read_csv(f"training/area/view_a/training_data_a_{n_frames}.csv")

test_df = pd.read_csv(f"testing/area/view_a/testing_data_a_{n_frames}.csv")

#, train_df_l, train_df_r
train_df = pd.concat([train_df_a], axis=0, ignore_index=True)

# file_name = f"training/combined_training_data_{n_frames}.csv"
# train_df.to_csv(file_name, index=False)
X_train = train_df_a.drop('label', axis=1)
y_train = train_df_a['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(
    kernel='rbf', 
    C=1.0, 
    gamma='scale', 
    random_state=8346
)

print("Treniranje SVM modela...")
svm_model.fit(X_train_scaled, y_train) 

y_pred = svm_model.predict(X_test_scaled)

print("\n--- REZULTATI (SVM) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nIzveštaj:")
print(classification_report(y_test, y_pred))