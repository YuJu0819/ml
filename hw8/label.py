import pandas as pd

pred = pd.read_csv('./prediction_a-3.csv', index_col=0)
idx = []
print(pred)
for i in range(len(pred)):
    if pred['score'][i]>=0.4:
        idx.append(i)
print(idx)