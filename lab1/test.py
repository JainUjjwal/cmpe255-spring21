import pandas as pd

data = {
    "order_id": ["12", "13", "14", "12", "13", "12"],
    "randstring": ["a", "b", "c", "d", "e", "f"],
    "sumstuff": [10, 12, 1, 2, 3, 4],
}

df = pd.DataFrame.from_dict(data)

print(df.head())

print(df.groupby(['order_id'],as_index=False))