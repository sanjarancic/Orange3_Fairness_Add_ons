from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data.table import Table
from Orange.widgets import settings
from imblearn.over_sampling import SMOTE
from Orange.data.pandas_compat import table_from_frame, table_to_frame
from Orange.widgets import gui, settings
from PyQt5.QtCore import Qt


class Owersampling(OWWidget):
    name = "Fair Oversampling"
    icon = "icons/oversampling_icon.png"

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
        oversampled_data = Output("Oversampled data", Table)

    def set_columns(self):
        if self.target is not None and self.sensitive is not None:
            target = self.columns[self.target]
            sensitive = self.columns[self.sensitive]
            self.oversample(target, sensitive)

    def oversample(self, target, sensitive):
        sm = SMOTE(random_state=0)

        df_train = table_to_frame(self.train_data)

        df_w = df_train.loc[df_train[sensitive] == 0]
        df_m = df_train.loc[df_train[sensitive] == 1]

        X_w = df_w.drop(target, axis=1)
        y_w = df_w[target]
        X_res_w, y_res_w = sm.fit_sample(X_w, y_w.ravel())
        df_w_smote = X_res_w.copy()
        df_w_smote[target] = y_res_w

        X_m = df_m.drop(target, axis=1)
        y_m = df_m[target]
        X_res_m, y_res_m = sm.fit_sample(X_m, y_m.ravel())
        df_m_smote = X_res_m.copy()
        df_m_smote[target] = y_res_m

        df_smote = df_w_smote.append(df_m_smote, ignore_index=True, sort=False)
        smote_table = table_from_frame(df_smote)
        self.Outputs.oversampled_data.send(smote_table)
