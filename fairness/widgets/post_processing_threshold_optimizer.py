from Orange.classification import LogisticRegressionLearner
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data.table import Table
from Orange.widgets import settings
from imblearn.over_sampling import SMOTE
from Orange.data.pandas_compat import table_from_frame, table_to_frame
from Orange.widgets import gui, settings
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
from scipy.optimize import minimize


class FairThresholdOptimizer(OWWidget):
    name = "Fair Threshold Optimizer"
    icon = "icons/threshold_optimizer_icon.png"

    upper_bounds = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
    upper_bound = settings.Setting(2)

    lower_bounds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    lower_bound = settings.Setting(8)

    want_main_area = False

    def __init__(self):
        super().__init__()
        # inputs
        self.s = None
        self.y = None
        self.y_hat = None

        self.box = gui.vBox(self.controlArea)

        self.upper_bound_combo = gui.comboBox(
            self.box, self, "upper_bound",
            items=[str(x) for x in self.upper_bounds], label="Upper bound: ",
            orientation=Qt.Horizontal, callback=self.threshold_optimizer)

        self.lower_bound_combo = gui.comboBox(
            self.box, self, "lower_bound", label="Lower bound: ",
            items=[str(x) for x in self.lower_bounds],
            orientation=Qt.Horizontal, callback=self.threshold_optimizer)

    class Inputs:
        s = Input("Sensitive attribute", Table)
        y = Input("Target attribute", Table)
        y_hat = Input("Predicted values", Table)

    @Inputs.s
    def set_s(self, s):
        """Set the input s."""
        self.s = s
        self.threshold_optimizer()

    @Inputs.y
    def set_y(self, y):
        """Set the input y."""
        self.y = y
        self.threshold_optimizer()

    @Inputs.y_hat
    def set_y_hat(self, y_hat):
        """Set the input y_hat."""
        self.y_hat = y_hat
        self.threshold_optimizer()

    class Outputs:
        threshold = Output("Threshold", Table)

    def threshold_optimizer(self):

        def F1(threshold, y, y_hat):
            y_pred = predict(threshold, y_hat)
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i, r in enumerate(y):
                if y_pred[i] == r and r == 1:
                    tp += 1
                elif y_pred[i] == r and r == 0:
                    tn += 1
                elif y_pred[i] != r and r == 1:
                    fn += 1
                else:
                    fp += 1

            precision = tp / (tp + fp)
            recall = tp / (tp + fp)

            return - 2 * precision * recall / (precision + recall)

        def disparate_impact(y_hat, s):
            return np.mean(y_hat[s == 0]) / np.mean(y_hat[s == 1])

        def predict(threshold, y_hat):
            predictions = []

            for pred in y_hat:
                if pred >= threshold:
                    predictions.append(1)
                else:
                    predictions.append(0)
            return pd.Series(predictions)

        def avg_disp_imp_upper(threshold, y_hat, s, upper):

            y_pred = predict(threshold, y_hat)

            r = upper - disparate_impact(y_pred, s)

            return r

        def avg_disp_imp_lower(threshold, y_hat, s, lower):

            y_pred = predict(threshold, y_hat)

            r = disparate_impact(y_pred, s) + lower

            return r

        if self.s is not None and self.y_hat is not None and self.y is not None:
            s = table_to_frame(self.s).iloc[:, 0]
            y = table_to_frame(self.y).iloc[:, 0]
            y_hat = table_to_frame(self.y_hat).iloc[:, 0]

            upper = self.upper_bounds[self.upper_bound]
            lower = self.lower_bounds[self.lower_bound]

            t = 0.5

            cons = ({'type': 'ineq', 'fun': avg_disp_imp_lower, 'args': (y_hat, s.values, lower)},
                    {'type': 'ineq', 'fun': avg_disp_imp_upper, 'args': (y_hat, s.values, upper)})

            model = minimize(fun=F1, x0=t, args=(y, y_hat),
                             method='SLSQP', constraints=cons)
            print(model.x)

            self.Outputs.threshold.send(table_from_frame(pd.DataFrame({'Threshold': model.x})))
