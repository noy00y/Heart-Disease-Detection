# Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings # Handling Warnings:
warnings.filterwarnings("ignore", category=FutureWarning) 
pd.options.mode.chained_assignment = None

# Class
class dataset():
    def __init__(self, file_path: str):
        self.dataset = pd.read_csv(file_path)
        return

    def __str__(self):

        return f'Head: \n {self.dataset.head()}\n Stats: \n{self.dataset.describe()}\n Columns: \n {self.dataset.columns}'

    def preprocess_dataset(self):
        # Renaming Columns for Readability:
        self.dataset.columns = ['age', 'sex', 'chest_pain_type', 'blood_pressure_resting', 'cholesterol', 'blood_suger_fasting',
               'electrocardiogram', 'max_heart_rate', 'angina', 's_trajectory', 'slope', 'blood_vessels', 'thalassemia', 'target']
        return

    def split_dataset_KNN(self, t_size: float, r_state: int):
        x = self.dataset.drop('target', axis=1)
        y = self.dataset['target']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=r_state)

        return x_train, x_test, y_train, y_test
    
    def split_datset_LR(self, t_size: float, r_state: int):
        x = self.dataset.drop('target', axis=1)
        y = self.dataset['target']

        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(x) 
        print(f'PCA Features: \n {pca_features}')

        x_train, x_test, y_train, y_test = train_test_split(pca_features, y, test_size=t_size, random_state=r_state)

        return x_train, x_test, y_train, y_test