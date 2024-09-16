import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('big_mart_train.csv')

print(data.sample(5))


