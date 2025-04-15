import pandas as pd

df = pd.read_csv('test_predictions.csv')

print(df['confidence'].mean())