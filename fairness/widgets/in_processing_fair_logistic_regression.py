from itertools import chain

from Orange.classification import LogisticRegressionLearner
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data.table import Table
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from PyQt5.QtCore import Qt
from Orange.widgets import gui, settings

from Orange.data.pandas_compat import table_from_frame, table_to_frame

from fairness.widgets.table_model import TableModel


class FairLogisticRegression(OWWidget):
    name = "Fair Logistic Regression"
    icon = "icons/fair_logistic_regression_icon.png"

    Thresholds = [-0.5, 0, 0.5]
    t = settings.Setting(1)

    penalty_types = ['Ridge', 'Lasso']
    penalty = settings.Setting(0)

    want_main_area = False

    C_s = list(chain(range(1000, 200, -50),
                     range(200, 100, -10),
                     range(100, 20, -5),
                     range(20, 0, -1),
                     [x / 10 for x in range(9, 2, -1)],
                     [x / 100 for x in range(20, 2, -1)],
                     [x / 1000 for x in range(20, 0, -1)]))

    C_index = settings.Setting(61)

    def __init__(self):
        super().__init__()
        # inputs
        self.y = None
        self.s = None
        self.X = None

        self.box = gui.widgetBox(self.controlArea, box=True)

        self.penalty_combo = gui.comboBox(
            self.box, self, "penalty", label="Regularization type: ",
            items=self.penalty_types, orientation=Qt.Horizontal,
            callback=self.zafar)

        gui.widgetLabel(self.box, "Strength:")

        self.box2 = gui.hBox(gui.indentedBox(self.box))
        gui.widgetLabel(self.box2, "Weak").setStyleSheet("margin-top:6px")
        self.c_slider = gui.hSlider(
            self.box2, self, "C_index", minValue=0, maxValue=len(self.C_s) - 1,
            callback=self.set_c, callback_finished=self.zafar, createLabel=False)
        gui.widgetLabel(self.box2, "Strong").setStyleSheet("margin-top:6px")
        self.box2 = gui.hBox(self.box)
        self.box2.layout().setAlignment(Qt.AlignCenter)
        self.c_label = gui.widgetLabel(self.box2)
        self.set_c()

        self.threshold_combo = gui.comboBox(
            self.box, self, "t", label="Threshold: ",
            items=[str(x) for x in self.Thresholds],
            orientation=Qt.Horizontal, callback=self.zafar)

    def set_c(self):
        alpha = self.C_s[self.C_index]
        fmt = "C={}" if alpha >= 1 else "C={:.3f}"
        self.c_label.setText(fmt.format(alpha))

    class Inputs:
        s = Input("Sensitive attribute", Table)
        y = Input("Target attribute", Table)
        X = Input("Feature attributes", Table)

    @Inputs.X
    def set_X(self, X):
        """Set the input X."""
        self.X = X
        self.zafar()

    @Inputs.s
    def set_s(self, s):
        """Set the input s."""
        self.s = s
        self.zafar()

    @Inputs.y
    def set_y(self, y):
        """Set the input y."""
        self.y = y
        self.zafar()

    class Outputs:
        coefficients = Output("Coefficients", Table, explicit=True)
        algorithm = Output('Algorithm', str)

    def sigmoid(self, w, X):
        return 1 / (1 + np.exp(-np.sum(w * X, axis=1)))

    def zafar(self, alpha=1.0):
        # https://www.jmlr.org/papers/volume20/18-262/18-262.pdf
        # Zafar, M. B., Valera, I., Gomez-Rodriguez, M., & Gummadi, K. P. (2019).
        # Fairness Constraints: A Flexible Approach for Fair Classification. J. Mach. Learn. Res., 20(75), 1-42.

        def logistic_loss(w, X, y):
            y_hat = self.sigmoid(w, X)

            return (-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

        # Ridge
        def logistic_loss_ridge(w, X, y, alpha):
            y_hat = self.sigmoid(w, X)

            return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat) + alpha * np.sum(w ** 2))

        # Lasso
        def logistic_loss_lasso(w, X, y, alpha):
            y_hat = self.sigmoid(w, X)

            return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat) + alpha * np.sum(np.abs(w)))

        def avg_disp_imp_upper(w, s, X, t):
            y_intensity = t - np.mean((s - np.mean(s)) * np.sum(w * X, axis=1))

            return y_intensity

        def avg_disp_imp_lower(w, s, X, t):
            y_intensity = np.mean((s - np.mean(s)) * np.sum(w * X, axis=1)) + t

            return y_intensity

        if self.s is not None and self.y is not None and self.X is not None:

            s = table_to_frame(self.s).iloc[:, 0]
            y = table_to_frame(self.y).iloc[:, 0]
            X = table_to_frame(self.X)

            t = self.Thresholds[self.t]
            penalty = self.penalty_types[self.penalty]
            w_0 = np.repeat(0, X.shape[1])
            alpha = self.C_s[self.C_index]

            cons = ({'type': 'ineq', 'fun': avg_disp_imp_lower, 'args': (s, X, t)},
                    {'type': 'ineq', 'fun': avg_disp_imp_upper, 'args': (s, X, t)})

            if penalty == 'l1':
                model = minimize(fun=logistic_loss_lasso, x0=w_0, args=(X, y, alpha),
                                 method='SLSQP', constraints=cons)
            elif penalty == 'l2':
                model = minimize(fun=logistic_loss_ridge, x0=w_0, args=(X, y, alpha),
                                 method='SLSQP', constraints=cons)
            else:
                model = minimize(fun=logistic_loss, x0=w_0, args=(X, y),
                                 method='SLSQP', constraints=cons)

            coef = table_from_frame(pd.DataFrame({'Coefficients': model.x}))
            self.Outputs.coefficients.send(coef)
            self.Outputs.algorithm.send('LogisticRegression')
