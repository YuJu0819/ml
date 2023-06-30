import pandas as pd

#df1 = pd.read_csv('80585-1.csv', index_col = 0)
#df2 = pd.read_csv('80585-2.csv', index_col = 0)
#df3 = pd.read_csv('predict_deep_transform_semi.csv', index_col = 0)
df1 = pd.read_csv('output.csv', index_col = 0)
df2 = pd.read_csv('output_liu.csv', index_col = 0)
# df3 = pd.read_csv('submission-78.csv', index_col = 0)
# df1 = pd.read_csv('final8086.csv', index_col = 0)
df3 = pd.read_csv('output-2.csv', index_col = 0)
# df3 = pd.read_csv('final809.csv', index_col = 0)
# df4 = pd.read_csv('final812.csv', index_col = 0)
# df5 = pd.read_csv('final811.csv', index_col = 0)
df_combine = pd.concat([df1, df2, df3],axis=1, )
df_combine = df_combine.mode(axis=1).dropna(axis=1)
# print(df_combine)

# df_combine = df_combine.astype('int32')
df_combine.columns = ['Category']
df_combine.to_csv('final.csv',index=True)