# Imports:
import matplotlib.pyplot as plt
from dataset import dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Class
class KNN:
    def __init__(self, dataset: dataset):
        self.x_train, self.x_test, self.y_train, self.y_test = dataset.split_dataset_KNN(.2, 42)
        self.model = KNeighborsClassifier(n_neighbors=10)
        self.best_params = None
        self.best_score = None

        return

    # Normal Fitting and Hyper Parameter fitting
    def fit(self, tune_fit: str):
        # Tuning with Hyper Parameters
        if tune_fit == "yes":
            param_grid = {
                'n_neighbors': [3, 5, 10, 15, 20],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
            grid_search = GridSearchCV(self.model, param_grid, cv = 5) # 5 folds
            grid_search.fit(self.x_train, self.y_train) # training the datasets on all the different hyper params
            self.best_params = grid_search.best_params_
            self.model = grid_search.best_estimator_ # setting model to best predictor
            self.best_score = grid_search.best_score_

            print(f"Best Param: {grid_search.best_params_}, Best Model: {grid_search.best_estimator_ }\n")

        else:
            self.model.fit(self.x_train, self.y_train)

        return

    def predict(self):
        return self.model.predict(self.x_test)

    def score(self):
        y_pred = self.predict() # get predictions
        c_matrix = confusion_matrix(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred) # gets accuracy of model by comparing actual y to y'
        report = classification_report(self.y_test, y_pred)

        return accuracy, report, c_matrix
    
    def plot_performance(self):
        y_pred = self.predict()

        # ROC curve:
        fpr, tpr, _ = roc_curve(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_pred)

        plt.plot(fpr, tpr, label='KNN (AUC = {:.2f})'.format(auc_score))
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Heart Disease Dataset')
        plt.legend()
        plt.show()

        return

class LR:
    def __init__(self, dataset: dataset):
        self.x_train, self.x_test, self.y_train, self.y_test = dataset.split_datset_LR(0.2, 42)
        self.model = LogisticRegression()

        return

    def fit(self):
        self.model.fit(self.x_train, self.y_train)
        return

    def predict(self):
        return self.model.predict(self.x_test)

    def score(self):
        y_pred = self.predict()
        c_matrix = confusion_matrix(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        return accuracy, report, c_matrix
    
    def plot_performance(self):
        y_pred = self.predict()

        # ROC curve:
        fpr, tpr, _ = roc_curve(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_pred)

        plt.plot(fpr, tpr, label='LR (AUC = {:.2f})'.format(auc_score))
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Heart Disease Dataset')
        plt.legend()
        plt.show()

        return