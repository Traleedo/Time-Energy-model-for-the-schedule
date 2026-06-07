import pandas as pd
df = pd.read_csv(r'benchmark_output\benchmark_cross_dataset.csv')
print(df["on_time_rate"])