import numpy as np
import pandas as pd

def most_label(lst):
    return max(set(lst), key=lst.count)


class KNearestNeighbour_algorithm:
    def __init__(self, n_neighbour: int = 5, distance_metric: str = "Euclidean"):
        self.n_neighbour = n_neighbour
        self.distance_metric = distance_metric
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def metrics(self, x, y):
        if self.distance_metric == "Euclidean":
            metric = np.sqrt(np.sum((x - y)**2))
        if self.distance_metric == "Manhattan":
            metric = np.sum(np.abs(x - y))
        return metric
    
    def get_neighbour(self, new_point):
        distances = []
        labels = []
        
        for i in range(len(self.x_train)):
            distances.append(self.metrics(new_point, self.x_train[i]))
            labels.append(self.y_train[i])

        # Sort lists based on distances
        sorted_data = sorted(zip(distances, labels), key = lambda x : x[0])

        # Unpack sorted data into separate lists
        sorted_distances, sorted_labels = zip(*sorted_data)
        neighbour_distances = list(sorted_distances)[:self.n_neighbour]
        neighbour_labels = list(sorted_labels)[:self.n_neighbour]
        return neighbour_distances, neighbour_labels

    def predict(self, x_test):
        prediction = []
        for i in range(len(self.x_test)):
            neighbour_distances = self.get_neighbour(x_test[i])[0]
            neighbour_labels = self.get_neighbour(x_test[i])[1]
            prediction.append(most_label(neighbour_labels))
        return prediction


class Evaluate:

    @staticmethod
    def accuracy(y_test, y_pred):
        correct_prediction = 0
        for i in range(len(y_test)):
            if y_test[i] == y_pred[i]:
                correct_prediction += 1
        accuracy_score = correct_prediction/ len(y_test)
        return accuracy_score
    
    @staticmethod
    def confusion_matrix(y_test, y_pred):
        classes = np.unique(y_test + y_pred)
        num_classes = len(classes)
        c_m = np.zeros((num_classes, num_classes), dtype = int)

        for i in range(num_classes):
                true_class = classes.searchsorted(y_test[i]) 
                pred_class = classes.searchsorted(y_pred[i])
                c_m[true_class, pred_class] += 1
        return c_m
    

    @staticmethod 
    def classifiication_report(y_test, y_pred):
        classes = np.unique(y_test + y_pred)
        num_classes = len(classes)
        c_m = np.zeros((num_classes, num_classes), dtype = int)

        for i in range(num_classes):
            true_class = classes.searchsorted(y_test[i]) 
            pred_class = classes.searchsorted(y_pred[i])
            c_m[true_class][pred_class] += 1
        
        c_report = pd.DataFrame(index = classes, columns = ["precision", "recall", "f1-score", "support"])
        for i in range(num_classes):
            # Precision
            c_report.loc[i, "precision"] = c_m[i, i] / np.sum(c_m[:, i]) 
            # Recall
            c_report.loc[i, "recall"] = c_m[i, i] / np.sum(c_m[i, :])
            # F1_score
            c_report.loc[i, "f1-score"] = 2 * ((c_report[i, 0] * c_report[i, 1])/ (c_report[i, 0] + c_report[i, 1]))
            # Support
            c_report.loc[i, "support"] = np.sum(c_m[i, :])

        return c_report

        






