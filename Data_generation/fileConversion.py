import pandas as pd

data = pd.read_excel("generated_binary_data.xlsx", index_col=None, header=None)
data = data.transpose()

data.to_csv('glass_artificial.csv', index = False) 

