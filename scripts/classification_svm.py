import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

n_frames = 5

train_df_a = pd.read_csv(f"training/point_to_center/view_a/training_data_a_{n_frames}.csv")
test_df = pd.read_csv(f"testing/point_to_center/view_a/testing_data_a_{n_frames}.csv")

X_train = train_df_a.drop('label', axis=1)
y_train = train_df_a['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

pca_svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)), 
    ('svm', SVC(
        kernel='rbf', 
        C=100.0,           
        gamma=0.01,        
        class_weight='balanced',
        random_state=8346
    ))
])

print("Treniranje PCA + SVM modela...")
pca_svm_pipeline.fit(X_train, y_train) 

y_pred = pca_svm_pipeline.predict(X_test)

print("\n--- REZULTATI (PCA + SVM) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

print("\nIzveštaj:")
print(classification_report(y_test, y_pred))