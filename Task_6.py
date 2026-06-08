import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# ==========================================
# 1. Data Loading & Normalization
# ==========================================
df = pd.read_csv('DATASET-balanced.csv')
df['LABEL'] = df['LABEL'].map({'REAL': 1, 'FAKE': 0}) # Encode Target

X = df.drop(columns=['LABEL'])
y = df['LABEL']

# Train-Test Split & Scaling (Crucial for KNN!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data Loaded & Normalized!")

# ==========================================
# 2. Experimenting with different values of K
# ==========================================
print("\n🔍 Finding the best K value...")
k_values = range(1, 20, 2) # Odd numbers to avoid ties
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_accuracies.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    test_accuracies.append(accuracy_score(y_test, knn.predict(X_test_scaled)))

# Plotting the K-Selection Graph (Elbow Curve)
plt.figure(figsize=(10, 5))
plt.plot(k_values, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(k_values, test_accuracies, label='Testing Accuracy', marker='o')
plt.title('KNN: Accuracy vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 3. Final Model Evaluation (Using K=3)
# ==========================================
best_k = 3 # Usually, 3 or 5 works best here
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
y_pred = final_knn.predict(X_test_scaled)

print(f"\n🏆 Final Model Performance (K={best_k}):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted FAKE', 'Predicted REAL'], 
            yticklabels=['Actual FAKE', 'Actual REAL'])
plt.title(f'Confusion Matrix (K={best_k})')
plt.show()

# ==========================================
# 4. Decision Boundary Visualization (Using PCA)
# ==========================================
# KNN ke 26D data ko visualize karna impossible hai, isliye PCA se isko 2D mein badal rahe hain
print("\n🎨 Generating Decision Boundary Visualization...")
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train a new KNN just for 2D visualization
knn_pca = KNeighborsClassifier(n_neighbors=best_k)
knn_pca.fit(X_train_pca, y_train)

# Create a mesh grid
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(10, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
# Scatter plot of test data
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
plt.title(f'KNN Decision Boundary (PCA Reduced to 2D) | K={best_k}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()