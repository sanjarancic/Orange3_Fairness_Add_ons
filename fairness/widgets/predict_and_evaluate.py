from itertools import chain

from Orange.classification import LogisticRegressionLearner
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data.table import Table
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QSplitter
from scipy.optimize import minimize
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from Orange.widgets import gui, settings
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from Orange.data.pandas_compat import table_from_frame, table_to_frame

from fairness.widgets.table_model import TableModel


class PredictEvaluate(OWWidget):
    name = "Predict and Evaluate"
    icon = "icons/predict.png"

    columns = ['Select column']
    target = settings.Setting(0)

    def __init__(self):
        super().__init__()
        # inputs
        self.algorithm = None
        self.coefficients = None
        self.test_data = None

        self.box = gui.vBox(self.controlArea)

        self.target_combo = gui.comboBox(
            self.box, self, "target",
            items=[col for col in self.columns], label="Target column: ",
            orientation=Qt.Horizontal, callback=self.set_columns)

        self.table = QtWidgets.QTableView()
        self.evaluations = QtWidgets.QTableView()
        self.mainArea.layout().addWidget(self.table)
        self.mainArea.layout().addWidget(self.evaluations)

    class Inputs:
        algorithm = Input("Algorithm", str)
        coefficients = Input("Coefficients", Table)
        test_data = Input("Test data", Table)

    @Inputs.algorithm
    def set_algorithm(self, algorithm):
        """Set the input algorithm."""
        self.algorithm = algorithm
        self.set_columns()

    @Inputs.coefficients
    def set_coefficients(self, coefficients):
        """Set the input coefficients."""
        self.coefficients = coefficients
        self.set_columns()

    @Inputs.test_data
    def set_test_data(self, test_data):
        """Set the input test_data."""
        self.test_data = test_data

        df_test = table_to_frame(self.test_data)
        model = TableModel(df_test)
        self.table.setModel(model)
        self.columns = list(df_test.columns)

        self.target_combo.clear()
        self.target_combo.addItems([col for col in df_test.columns])
        self.set_columns()

    def set_columns(self):
        if self.target is not None:
            target = self.columns[self.target]
            self.predict(target)

    class Outputs:
        predictions = Output("Predictions", Table, explicit=True)


    def sigmoid(self, w, X):
        return 1 / (1 + np.exp(-np.sum(w * X, axis=1)))

    def predict(self, target):
        test_data = table_to_frame(self.test_data)
        y_test = test_data[target].astype('int')
        X_test = test_data.drop(target, axis=1)

        if self.test_data is not None and self.algorithm is not None and self.coefficients is not None:
            if self.algorithm == 'LogisticRegression' or  self.algorithm == 'Reweighting':
                predictions_proba = self.sigmoid(self.coefficients, X_test)
                predictions = [1 if predictions_proba[i]>=0.5 else 0 for i in range(len(predictions_proba))]
                predictions = pd.Series(predictions)
                test_data['y_proba'] = predictions_proba
                test_data['y_pred'] = predictions
                model = TableModel(test_data)
                self.table.setModel(model)
            elif self.algorithm == 'SVM':
                predictions_proba = np.sign(np.dot(X_test, self.coefficients))
                predictions = [1 if predictions_proba[i] > 0 else 0 for i in range(len(predictions_proba))]
                predictions = pd.Series(predictions)
                test_data['y_pred'] = predictions
                model = TableModel(test_data)
                self.table.setModel(model)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            evaluations = pd.DataFrame([
                [accuracy, recall, precision, f1],
            ], columns=['Accuracy', 'Recall', 'Precision', 'F1'])

            model = TableModel(evaluations)
            self.evaluations.setModel(model)
            test_data_table = table_from_frame(test_data)
            self.Outputs.predictions.send(test_data_table)
