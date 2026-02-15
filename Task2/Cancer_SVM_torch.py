
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv("data/gene_expression.csv")

X = df.drop("Cancer Present", axis=1).values
y = df["Cancer Present"].values

y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(

)

#incomplete code    