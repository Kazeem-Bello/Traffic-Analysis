import numpy as np
import pandas as pd

def most_label(lst):
    return max(set(lst), key=lst.count)


class KNearestNeighbour_algorithm:
    def __init__(self, n_neighbour: int = 5, distance_metric: str = "Euclidean"):
        self.n_neighbour = n_neighbour
        self.distance_metric = distance_metric
    
    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

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
        _, sorted_labels = zip(*sorted_data)
        neighbour_labels = list(sorted_labels)[:self.n_neighbour]
        return neighbour_labels

    def predict(self, x_test):
        prediction = []
        for i in range(len(x_test)):
            # neighbour_distances = self.get_neighbour(x_test[i])[0]
            neighbour_labels = self.get_neighbour(x_test[i])
            prediction.append(most_label(neighbour_labels))
        return prediction


class evaluation_metrics:

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
        classes = np.unique(np.concatenate((y_pred, y_test)))
        num_classes = len(classes)
        c_m = np.zeros((num_classes, num_classes), dtype = int)

        for i in range(len(classes)):
            for j in range(len(classes)):
                c_m[i,j] = np.sum((y_test == classes[i]) & (y_pred == classes[j]))
        return c_m
    

    @staticmethod 
    def classification_report(y_test, y_pred):
        classes =  np.unique(np.concatenate((y_pred, y_test)))
        num_classes = len(classes)
        c_m = np.zeros((num_classes, num_classes), dtype = int)
        c_report = pd.DataFrame(columns = ["precision", "recall", "f1-score", "support"], index = classes)

        for i in range(len(classes)):
            for j in range(len(classes)):
                c_m[i,j] = np.sum((y_test == classes[i]) & (y_pred == classes[j]))

        for i in range(num_classes):
            # Precision
            precision  = c_m[i, i] / np.sum(c_m[:, i]) if np.sum(c_m[:, i]) != 0 else 0
            c_report.loc[i, "precision"] = "{:.2f}".format(precision)
            # Recall
            recall = c_m[i, i] / np.sum(c_m[i, :]) if np.sum(c_m[i, :]) != 0 else 0
            c_report.loc[i, "recall"] = "{:.2f}".format(recall)
            # F1_score
            f1_score = (2 * precision * recall)/ (precision + recall) if (precision + recall) != 0 else 0
            c_report.loc[i, "f1-score"] = "{:.2f}".format(f1_score)
            # Support
            c_report.loc[i, "support"] = np.sum(c_m[i, :])

        return c_report

        






