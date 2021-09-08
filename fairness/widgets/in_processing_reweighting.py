from Orange.classification import LogisticRegressionLearner
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data.table import Table
from Orange.data.pandas_compat import table_from_frame, table_to_frame
import pandas as pd
from PyQt5.QtCore import Qt
from Orange.widgets import gui, settings
from sklearn.linear_model import LogisticRegression


class FairReweighting(OWWidget):
    name = "Fair Reweighting"
    icon = "icons/reweighting_icon.png"

    columns = ['Select column']
    target = settings.Setting(0)
    sensitive = settings.Setting(0)

    want_main_area = False

    def __init__(self):
        super().__init__()
        # inputs
        self.train_data = None
        self.box = gui.vBox(self.controlArea)

        self.target_combo = gui.comboBox(
            self.box, self, "target",
            items=[col for col in self.columns], label="Target column: ",
            orientation=Qt.Horizontal, callback=self.set_columns)

        self.sensitive_combo = gui.comboBox(
            self.box, self, "sensitive", label="Sensitive column: ",
            items=[col for col in self.columns],
            orientation=Qt.Horizontal, callback=self.set_columns)

    class Inputs:
        train_data = Input("Train data", Table)

    @Inputs.train_data
    def set_train_data(self, train_data):
        """Set the input train_data."""
        self.train_data = train_data

        df_train = table_to_frame(self.train_data)
        self.columns = list(df_train.columns)

        self.target_combo.clear()
        self.target_combo.addItems([col for col in df_train.columns])

        self.sensitive_combo.clear()
        self.sensitive_combo.addItems([col for col in df_train.columns])

    class Outputs:
        coefficients = Output("Coefficients", Table)

    def set_columns(self):
        if self.target is not None and self.sensitive is not None:
            target = self.columns[self.target]
            sensitive = self.columns[self.sensitive]
            self.reweight(target, sensitive)

    def reweight(self, target, sensitive):

        df_train = table_to_frame(self.train_data)

        y_train = df_train[target]
        X_train = df_train.drop(target, axis=1)
        s_train = X_train[sensitive]

        n_s0 = pd.crosstab(y_train, s_train)[0][0] + pd.crosstab(y_train, s_train)[0][1]
        n_s1 = pd.crosstab(y_train, s_train)[1][0] + pd.crosstab(y_train, s_train)[1][1]
        n_y0 = pd.crosstab(y_train, s_train)[0][0] + pd.crosstab(y_train, s_train)[1][0]
        n_y1 = pd.crosstab(y_train, s_train)[0][1] + pd.crosstab(y_train, s_train)[1][1]

        n_samples = s_train.shape

        weights = []
        weights.append(n_samples[0] / (n_s0 * n_y0))
        weights.append(n_samples[0] / (n_s1 * n_y0))
        weights.append(n_samples[0] / (n_s0 * n_y1))
        weights.append(n_samples[0] / (n_s1 * n_y1))

        sy = pd.concat([s_train, y_train], axis=1).reset_index()
        sy.drop('index', inplace=True, axis=1)

        new_weights = []
        for index, row in sy.iterrows():
            if int(row[sensitive]) == 0 and int(row[target]) == 0:
                new_weights.append(weights[0])
            elif int(row[sensitive]) == 1 and int(row[target]) == 0:
                new_weights.append(weights[1])
            elif int(row[sensitive]) == 0 and int(row[target]) == 1:
                new_weights.append(weights[2])
            elif int(row[sensitive]) == 1 and int(row[target]) == 1:
                new_weights.append(weights[3])

        model_lg = LogisticRegression()
        model_lg.fit(X_train, y_train, sample_weight=new_weights)
        y_pred_proba = table_from_frame(pd.DataFrame({'Coefficients': model_lg.predict_proba(X_train)[:,1]}))
        self.Outputs.coefficients.send(y_pred_proba)
