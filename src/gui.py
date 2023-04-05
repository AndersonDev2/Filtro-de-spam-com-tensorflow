from PyQt6 import QtWidgets, uic
import handle_data
import pandas as pd
import random
import os
import sys
import nn
import numpy as np

os.system("cls")
base_dir = os.path.dirname(__file__)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.model = nn.Model()
        uic.loadUi(os.path.join(base_dir, "gui.ui"), self)
        self.setWindowTitle("Classificação de texto")
        self.RecarregarButton.clicked.connect(self.recarregar)
        self.pegar_texto_aleatorio()
        self.predizer()
        self.show()

    def recarregar(self):
        self.pegar_texto_aleatorio()
        self.predizer()
        return

    def pegar_texto_aleatorio(self):
        test_df = pd.read_csv(os.path.join(base_dir, "dataset/test_df.csv"))
        random_input = test_df.iloc[random.randint(0, len(test_df)-1)]
        texto = random_input["v2"]
        x = handle_data.normalizar_texto(texto)
        y = str(False) if int(random_input["v1"] == 0) else str(True)
        self.Texto.setPlainText(str(texto))
        self.RealLabel.setText(y)

    def predizer(self):
        texto = self.Texto.toPlainText()
        n_texto = handle_data.normalizar_texto(texto)
        resultado = self.model.predict(np.array(n_texto)).item()
        r = str(False) if (resultado < 0.5) else str(True)
        self.ModelLabel.setText(r)
        return


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
app.exec()
