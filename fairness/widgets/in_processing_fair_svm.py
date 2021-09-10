from Orange.classification import LogisticRegressionLearner
from Orange.data import table_to_frame
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data.table import Table
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from PyQt5.QtCore import Qt
from Orange.widgets import gui, settings

from fairness.widgets.table_model import TableModel


class FairSVM(OWWidget):
    name = "Fair SVM"
    icon = "icons/fair_svm_icon.png"

    Thresholds = [-0.5, 0, 0.5]
    t = settings.Setting(1)

    want_main_area = False

    def __init__(self):
        super().__init__()
        # inputs
        self.y = None
        self.s = None
        self.y_pred = None
        self.X = None

        self.box = gui.vBox(self.controlArea, "Threshold")

        gui.comboBox(
            self.box, self, "t",
            items=[str(x) for x in self.Thresholds],
            orientation=Qt.Horizontal, callback=self.zafar_svm)

    class Inputs:
        s = Input("Sensitive attribute", Table)
        y = Input("Target attribute", Table)
        X = Input("Feature attributes", Table)

    @Inputs.X
    def set_X(self, X):
        """Set the input X."""
        self.X = X
        self.zafar_svm()

    @Inputs.s
    def set_s(self, s):
        """Set the input s."""
        self.s = s
        self.zafar_svm()

    @Inputs.y
    def set_y(self, y):
        """Set the input y."""
        self.y = y
        self.zafar_svm()

    class Outputs:
        coefficients = Output("Coefficients", Table, explicit=True)
        algorithm = Output('Algorithm', str)

    def zafar_svm(self):
        def hinge_loss(w, X, y):
            yz = y * np.dot(X, w)
            yz = np.maximum(np.zeros_like(yz), (1 - yz))  # hinge function

            return np.mean(yz)

        def avg_disp_imp_upper(w, X_0, X_1, y_0, y_1, z_1, z_0, t):
            y_intensity = t + z_1 / z * hinge_loss(w, X_1, y_1) - z_0 / z * hinge_loss(w, X_0, y_0)

            return y_intensity

        def avg_disp_imp_lower(w, X_0, X_1, y_0, y_1, z_1, z_0, t):
            y_intensity = -z_1 / z * hinge_loss(w, X_1, y_1) + z_0 / z * hinge_loss(w, X_0, y_0) + t

            return y_intensity
        if self.s is not None and self.y is not None and self.X is not None:

            s = table_to_frame(self.s).iloc[:, 0]
            y = table_to_frame(self.y).iloc[:, 0]
            X = table_to_frame(self.X)

            z_1 = len(s[s == 1])
            z_0 = len(s[s == 0])
            z = len(s)

            y_0 = y[s == 0]
            y_1 = y[s == 1]

            X_0 = X[s == 0]
            X_1 = X[s == 1]

            w_0 = np.repeat(0, X.shape[1])

            t = self.Thresholds[self.t]

            cons = ({'type': 'ineq', 'fun': avg_disp_imp_lower, 'args': (X_0, X_1, y_0, y_1, z_1, z_0, t)},
                    {'type': 'ineq', 'fun': avg_disp_imp_upper, 'args': (X_0, X_1, y_0, y_1, z_1, z_0, t)})

            model = minimize(fun=hinge_loss, x0=w_0, args=(X, y),
                             method='SLSQP', constraints=cons)

            self.Outputs.coefficients.send(model.x)
            self.Outputs.algorithm.send('SVM')