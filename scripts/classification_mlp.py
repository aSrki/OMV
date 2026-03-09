from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

n_frames = 3

# Load data
train_df = pd.read_csv(f"training/area/view_a/training_data_a_{n_frames}.csv")
test_df = pd.read_csv(f"testing/area/view_a/testing_data_a_{n_frames}.csv")

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 1. Scaling - Essential for Neural Networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Initialize MLP Classifier
# This setup uses two hidden layers (64 neurons and 32 neurons)
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 64, 32, 16, 8), 
    activation='relu', 
    solver='adam', 
    max_iter=1000,
    random_state=8346,
    verbose=False # Set to True to see the loss decrease during training
)

print("Treniranje MLP (Neural Network) modela...")
mlp.fit(X_train_scaled, y_train) 

# 3. Prediction
y_pred = mlp.predict(X_test_scaled)

print("\n--- REZULTATI (MLP) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nIzveštaj:")
print(classification_report(y_test, y_pred))