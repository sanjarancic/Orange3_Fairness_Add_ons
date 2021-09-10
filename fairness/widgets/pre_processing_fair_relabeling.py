from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data.table import Table
from Orange.widgets import settings
from imblearn.over_sampling import SMOTE
from Orange.data.pandas_compat import table_from_frame, table_to_frame
from Orange.widgets import gui, settings
from PyQt5.QtCore import Qt
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd


class FairRelabeling(OWWidget):
    name = "Fair Relabeling"
    icon = "icons/relabel-icon.png"

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
        relabeled_data = Output("Relabeled data", Table)

    def set_columns(self):
        if self.target is not None and self.sensitive is not None:
            target = self.columns[self.target]
            sensitive = self.columns[self.sensitive]
            self.relabel(target, sensitive)

    def relabel(self, target, sensitive):
        df_train = table_to_frame(self.train_data)
        y_train = df_train[target]
        X_train = df_train.drop(target, axis=1)

        model_lg = LogisticRegression()
        model_lg.fit(X_train, y_train)
        y_pred_proba = model_lg.predict_proba(X_train)

        df_train['y_pred_proba'] = y_pred_proba[:, 1]

        df_train_1 = df_train[df_train[sensitive] == 1]
        df_train_0 = df_train[df_train[sensitive] == 0]

        z_0 = df_train_0.shape[0]
        z_1 = df_train_1.shape[0]
        z_1_y_1 = df_train_1[df_train_1[target] == '1'].shape[0]
        z_0_y_1 = df_train_0[df_train_0[target] == '1'].shape[0]

        M = np.round(np.abs((z_0 * z_1_y_1 - z_1 * z_0_y_1) / (z_0 + z_1)))
        print(M)

        a = df_train_1.sort_values(by='y_pred_proba')
        b = df_train_0.sort_values(by='y_pred_proba')

        print('a')
        print(a)

        print('b')
        print(b)

        index_a = 0
        new_y_a = []
        for i, row in a.iterrows():
            if index_a < M:
                if row[target] == '0':
                    new_y_a.append(1)
                else:
                    new_y_a.append(0)
            else:
                new_y_a.append(row[target])
            index_a += 1
        a[target] = new_y_a

        print(a)

        index_b = 0
        new_y_b = []
        for i, row in b.iterrows():
            if index_b < M:
                if row[target] == '0':
                    new_y_b.append(1)
                else:
                    new_y_b.append(0)
            else:
                new_y_b.append(row[target])
            index_b += 1
        b[target] = new_y_b

        print(b)

        df_all = pd.concat([a, b])
        relabeled_table = table_from_frame(df_all)

        self.Outputs.relabeled_data.send(relabeled_table)
