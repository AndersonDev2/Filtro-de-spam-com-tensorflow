import os
import handle_data
import nn
import numpy as np
import pandas as pd
import random
from decimal import Decimal, ROUND_HALF_UP
os.system("cls")

base_dir = os.path.dirname(__file__)
model = nn.Model()


def test():
    test_df = pd.read_csv(os.path.join(base_dir, "dataset/test_df.csv"))
    random_input = test_df.iloc[random.randint(0, len(test_df)-1)]
    x = handle_data.normalizar_texto(random_input['Body'])
    y = int(random_input['Label'])
    result = model.predict(np.array(x)).item()
    result = 0 if (result < 0.5) else 1
    print(f"Resultado do modelo: {result}")
    print(f"Resultado Real: {y}")


def train():
    train_dataset, test_dataset = handle_data.get_dataset()
    x_train, y_train = train_dataset
    x_test, y_test = test_dataset
    model.fit(np.array(x_train), np.array(y_train),
              np.array(x_test), np.array(y_test), epochs=5)


def evaluate():
    train_dataset, test_dataset = handle_data.get_dataset()
    x_test, y_test = test_dataset
    model.model.evaluate(np.array(x_test), np.array(y_test))
