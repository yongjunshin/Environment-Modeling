import numpy as np
import pandas as pd
import plotly.express as px


df = pd.read_csv("data/ver1_fixed_interval/ver1_ft_60_10.csv", index_col='time')
fig1 = px.scatter(df[:1000], y=' color')
fig1.show()

np_arr = df.to_numpy()
color_np_arr = np_arr[300:400, 0]
print(color_np_arr)

new_df = pd.DataFrame({'-2':color_np_arr[:-2], '-1':color_np_arr[1:-1], '0':color_np_arr[2:]})

fig = px.scatter_3d(new_df, x='-2', y='-1', z='0', size_max=1)
fig.show()
