import pandas as pd

data = pd.read_excel("../data/generated_binary_data.xlsx", index_col=None, header=None)
data = data.transpose()

data.to_csv('../data/glass_artificial.csv', index = False) 

