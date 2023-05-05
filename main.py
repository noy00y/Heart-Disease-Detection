from models import KNN, LR
from dataset import dataset
from sklearn.preprocessing import StandardScaler

"""
Testing KNN:
- Hyper Parameters increases accuracy by 10%
- Scaling decreases accuracy but makes the model more stable to untested data
"""
# Creating Dataset
print("Testing KNN:")
print("------------")
dataset1 = dataset("src\heart.csv")
dataset1.preprocess_dataset()

# Create Model:
knn_model = KNN(dataset1)
scaler = StandardScaler()

# Scaling reduces performance but makes model more stable
knn_model.x_train = scaler.fit_transform(knn_model.x_train)
knn_model.x_test = scaler.fit_transform(knn_model.x_test)

knn_model.fit("yes") # hypertuning resulting in 90+ accuracy
accuracy1, report1, c_matrix1 = knn_model.score()

# Results:
print(f'Accuracy: \n{accuracy1}')
print(f'Report: \n{report1}')
print(f'Confusion Matrix: \n{c_matrix1}')
# knn_model.plot_performance() # plotting roc curve

# Feature Selection:
"""
Testing Logistic Regression:
"""
print("\nTesting LR:")
print("-----------")
dataset2 = dataset("src\heart.csv")
dataset2.preprocess_dataset()

# Create Model:
lr_model = LR(dataset2)

lr_model.fit()
accuracy2, report2, c_matrix2 = lr_model.score()

# Results:
print(f'Accuracy: \n{accuracy2}')
print(f'Report: \n{report2}')
print(f'Confusion Matrix: \n{c_matrix2}')
# lr_model.plot_performance()