import pandas as pd
import numpy as np

from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input
from Orange.data.table import Table
from Orange.data.pandas_compat import table_to_frame
from PyQt5 import  QtWidgets
from PyQt5.QtCore import Qt

from fairness.widgets.table_model import TableModel


class MeasureFairness(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "Measure Fairness"
    icon = "icons/equality.png"

    NNeighbors = [1, 3, 5, 7, 9, 11, 13]
    n_neighbors = settings.Setting(2)

    # want_main_area = False

    def __init__(self):
        super().__init__()
        # inputs
        self.y = None
        self.s = None
        self.y_pred = None
        self.X = None

        # table with results
        self.table = QtWidgets.QTableView()

        # metrics
        self.DI = None
        self.EOO = None
        self.PE = None
        self.PP = None
        self.C = None

        self.mainArea.layout().addWidget(self.table)

        self.box = gui.vBox(self.controlArea, "Number of neighbors")

        gui.comboBox(
            self.box, self, "n_neighbors",
            items=[str(x) for x in self.NNeighbors],
            orientation=Qt.Horizontal, callback=self.set_data)

    class Inputs:
        s = Input("Sensitive attribute", Table)
        y = Input("Target attribute", Table)
        y_pred = Input("Predicted values", Table)
        X = Input("Feature attributes", Table)

    @Inputs.X
    def set_X(self, X):
        """Set the input X."""
        self.X = X
        self.set_data()

    @Inputs.s
    def set_s(self, s):
        """Set the input s."""
        self.s = s
        self.set_data()

    @Inputs.y
    def set_y(self, y):
        """Set the input y."""
        self.y = y
        self.set_data()

    @Inputs.y_pred
    def set_y_pred(self, y_pred):
        "Set the input y_pred"
        self.y_pred = y_pred
        self.set_data()

    def set_data(self):
        if self.s is not None and self.y is not None and self.y_pred is not None and self.X is not None:
            n = self.NNeighbors[self.n_neighbors]
            self.DI = self.calculate_disparate_impact(self.s, self.y)
            self.EOO = self.calculate_equality_of_odds(self.y_pred, self.y, self.s, 1)
            self.PE = self.calculate_equality_of_odds(self.y_pred, self.y, self.s, 0)
            self.PP = self.calculate_preedictive_parity(self.y_pred, self.y, self.s)
            self.C = self.calculate_consistency(self.X, self.y_pred, n)

            data = pd.DataFrame([
                [self.C, self.DI, self.EOO, self.PE, self.PP],
            ], columns=['Consistency', 'Disparate Impact', 'Equality Of Opportunity', 'Predictive Equality',
                        'Predictive Parity'])

            model = TableModel(data)
            self.table.setModel(model)
        elif self.s is not None and self.y is not None and self.y_pred is not None:
            self.DI = self.calculate_disparate_impact(self.s, self.y)
            self.EOO = self.calculate_equality_of_odds(self.y_pred, self.y, self.s, 1)
            self.PE = self.calculate_equality_of_odds(self.y_pred, self.y, self.s, 0)
            self.PP = self.calculate_preedictive_parity(self.y_pred, self.y, self.s)

            data = pd.DataFrame([
                [self.DI, self.EOO, self.PE, self.PP],
            ], columns=['Disparate Impact', 'Equality Of Opportunity', 'Predictive Equality', 'Predictive Parity'])

            model = TableModel(data)
            self.table.setModel(model)
        elif self.s is not None and self.y is not None:
            self.DI = self.calculate_disparate_impact(self.s, self.y)
            data = pd.DataFrame([
                [self.DI],
            ], columns=['Disparate Impact'])

            model = TableModel(data)
            self.table.setModel(model)
        else:
            pass

    def calculate_disparate_impact(self, s, y):
        s = table_to_frame(s).iloc[:, 0]
        y = table_to_frame(y).iloc[:, 0]

        return np.mean(y[s == 0]) / np.mean(y[s == 1])

    def calculate_equality_of_odds(self, y_hat, y, s, out=1):
        s = table_to_frame(s).iloc[:, 0]
        y = table_to_frame(y).iloc[:, 0]
        y_hat = table_to_frame(y_hat).iloc[:, 0]

        eoo = np.mean(y_hat.values[(s.values == 1) & (y.values == out)]) / np.mean(
            y_hat.values[(s.values == 0) & (y.values == out)])

        return eoo

    def calculate_preedictive_parity(self, y_hat, y, s):
        s = table_to_frame(s).iloc[:, 0]
        y = table_to_frame(y).iloc[:, 0]
        y_hat = table_to_frame(y_hat).iloc[:, 0]

        pp = np.mean(y.values[(s.values == 1) & (y_hat.values == 1)]) / np.mean(
            y.values[(s.values == 0) & (y_hat.values == 1)])

        return pp

    def calculate_consistency(self, X, y_hat, n_neighbors):
        y_hat = table_to_frame(y_hat).iloc[:, 0]
        X = table_to_frame(X)

        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors)
        neigh.fit(X)
        _, indices = neigh.kneighbors(X)

        sum = 0
        for i in range(len(y_hat)):
            sum += y_hat[i] - np.mean(y_hat[indices[i]])
        consistency = 1 - sum / len(y_hat)

        return consistency
