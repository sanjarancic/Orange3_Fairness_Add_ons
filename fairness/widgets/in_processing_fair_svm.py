from Orange.classification import LogisticRegressionLearner
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data.table import Table
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from PyQt5.QtCore import Qt
from Orange.widgets import gui, settings

from fairness.widgets.table_model import TableModel


class FairLogisticRegression(OWWidget):
    name = "Fair SVM"
    icon = "icons/fair_svm_icon.png"

    T = [-0.5, 0, 0.5]
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
            items=[str(x) for x in self.T],
            orientation=Qt.Horizontal, callback=self.set_data)

    class Inputs:
        s = Input("Sensitive attribute", Table)
        y = Input("Target attribute", Table)
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

    class Outputs:
        sample = Output("Sampled Data", Table)

    def set_data(self):
        if self.s is not None and self.y is not None and self.y_pred is not None and self.X is not None:
            t = self.T[self.t]
            self.DI = self.calculate_disparate_impact(self.s, self.y)

            data = pd.DataFrame([
                [self.C, self.DI, self.EOO, self.PE, self.PP],
            ], columns=['Consistency', 'Disparate Impact', 'Equality Of Opportunity', 'Predictive Equality',
                        'Predictive Parity'])

            model = TableModel(data)
            self.table.setModel(model)
        else:
            pass

    def sigmoid(self, w, X):
        return 1 / (1 + np.exp(-np.sum(w * X, axis=1)))

    def zafar(self, X, y, s, t=0, alpha=1.0, penalty='l2'):
        # https://www.jmlr.org/papers/volume20/18-262/18-262.pdf
        # Zafar, M. B., Valera, I., Gomez-Rodriguez, M., & Gummadi, K. P. (2019).
        # Fairness Constraints: A Flexible Approach for Fair Classification. J. Mach. Learn. Res., 20(75), 1-42.

        def logistic_loss(w, X, y):
            y_hat = self.sigmoid(w, X)

            return (-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

        # Ridge
        def logistic_loss_ridge(w, X, y):
            y_hat = self.sigmoid(w, X)

            return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat) + alpha * np.sum(w ** 2))

        # Lasso
        def logistic_loss_lasso(w, X, y):
            y_hat = self.sigmoid(w, X)

            return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat) + alpha * np.sum(np.abs(w)))

        def avg_disp_imp_upper(w, s, X, t):
            y_intensity = t - np.mean((s - np.mean(s)) * np.sum(w * X, axis=1))

            return y_intensity

        def avg_disp_imp_lower(w, s, X, t):
            y_intensity = np.mean((s - np.mean(s)) * np.sum(w * X, axis=1)) + t

            return y_intensity

        w_0 = np.repeat(0, X.shape[1])

        cons = ({'type': 'ineq', 'fun': avg_disp_imp_lower, 'args': (s, X, t)},
                {'type': 'ineq', 'fun': avg_disp_imp_upper, 'args': (s, X, t)})

        if penalty == 'l1':
            model = minimize(fun=logistic_loss_lasso, x0=w_0, args=(X, y),
                             method='SLSQP', constraints=cons)
        elif self.penalty == 'l2':
            model = minimize(fun=logistic_loss_ridge, x0=w_0, args=(X, y),
                             method='SLSQP', constraints=cons)
        else:
            model = minimize(fun=logistic_loss, x0=w_0, args=(X, y),
                             method='SLSQP', constraints=cons)

        return model.x
