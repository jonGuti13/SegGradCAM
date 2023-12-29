import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def mySortFcn(s):
    init = s.find("_") + 1
    fin = init + s[s.find("_") + 1:].find("_")
    id = s[init:fin]

    return int(str(id).zfill(2))

listado = glob.glob("/workspace/*info*.npy")
listado = sorted(listado, key=mySortFcn)

df = pd.DataFrame(columns = ['Img_num', 'Layer_num', 'is_eff', 'center_coor_x', 'center_coor_y', 'radius_size'])

for file in listado:
    mini_info = np.load(file) 
    id = str(mySortFcn(file)).zfill(2)

    df = df._append({'Img_num' : file[-7:-4], 'Layer_num' : id, 'is_eff' : 0, 'center_coor_x' : mini_info[0, 0], 'center_coor_y' : mini_info[0, 1], 'radius_size' : mini_info[0, 2]}, ignore_index = True)
    df = df._append({'Img_num' : file[-7:-4], 'Layer_num' : id, 'is_eff' : 1, 'center_coor_x' : mini_info[1, 0], 'center_coor_y' : mini_info[1, 1], 'radius_size' : mini_info[1, 2] + np.sqrt((mini_info[1, 0] - mini_info[0, 0])**2 + (mini_info[1, 1] - mini_info[0, 1])**2) }, ignore_index = True)

#+ np.sqrt((mini_info[1, 0] - mini_info[0, 0])**2 + (mini_info[1, 1] - mini_info[0, 1])**2)

df_is_eff = df[df["is_eff"] == 1]
df_no_eff = df[df["is_eff"] == 0]

cut = 0
bw_adjust = 0.25

fig, ax = plt.subplots(figsize=(10*2, 5*2))
sns.catplot(data=df_is_eff, x="radius_size",  y="Layer_num", kind="violin", inner="quartile", cut = cut, bw_adjust = bw_adjust)
plt.savefig('/workspace/is_eff_violin_plot_radius_size_layer_num.png')

df_is_eff_conv_2d_1 = df_is_eff[df_is_eff["Layer_num"] == "01"]
fig, ax = plt.subplots(figsize=(10*2, 5*2))
sns.catplot(data=df_is_eff_conv_2d_1, x="radius_size",  kind="violin", inner="quartile", cut = cut, bw_adjust = bw_adjust)
plt.xlabel('')
plt.xticks(fontsize=16)
plt.savefig('/workspace/is_eff_conv2d_1_violin_plot_radius_size.png')

df_is_eff_conv_2d_3 = df_is_eff[df_is_eff["Layer_num"] == "03"]
fig, ax = plt.subplots(figsize=(10*2, 5*2))
sns.catplot(data=df_is_eff_conv_2d_3, x="radius_size",  kind="violin", inner="quartile", cut = cut, bw_adjust = bw_adjust)
plt.xlabel('')
plt.xticks(fontsize=16)
plt.savefig('/workspace/is_eff_conv2d_3_violin_plot_radius_size.png')

df_is_eff_conv_2d_5 = df_is_eff[df_is_eff["Layer_num"] == "05"]
fig, ax = plt.subplots(figsize=(10*2, 5*2))
sns.catplot(data=df_is_eff_conv_2d_5, x="radius_size",  kind="violin", inner="quartile", cut = cut, bw_adjust = bw_adjust)
plt.xlabel('')
plt.xticks(fontsize=16)
plt.savefig('/workspace/is_eff_conv2d_5_violin_plot_radius_size.png')

df_is_eff_conv_2d_7 = df_is_eff[df_is_eff["Layer_num"] == "07"]
fig, ax = plt.subplots(figsize=(10*2, 5*2))
sns.catplot(data=df_is_eff_conv_2d_7, x="radius_size",  kind="violin", inner="quartile", cut = cut, bw_adjust = bw_adjust)
plt.xlabel('')
plt.xticks(fontsize=16)
plt.savefig('/workspace/is_eff_conv2d_7_violin_plot_radius_size.png')

df_is_eff_conv_2d_15 = df_is_eff[df_is_eff["Layer_num"] == "15"]
fig, ax = plt.subplots(figsize=(10*2, 5*2))
sns.catplot(data=df_is_eff_conv_2d_15, x="radius_size",  kind="violin", inner="quartile", cut = cut, bw_adjust = bw_adjust)
plt.xlabel('')
plt.xticks(fontsize=16)
plt.savefig('/workspace/is_eff_conv2d_15_violin_plot_radius_size.png')

df_is_eff_conv_2d_17 = df_is_eff[df_is_eff["Layer_num"] == "17"]
fig, ax = plt.subplots(figsize=(10*2, 5*2))
sns.catplot(data=df_is_eff_conv_2d_17, x="radius_size",  kind="violin", inner="quartile", cut = cut, bw_adjust = bw_adjust)
plt.xlabel('')
plt.xticks(fontsize=16)
plt.savefig('/workspace/is_eff_conv2d_17_violin_plot_radius_size.png')

df_is_eff_conv_2d_19 = df_is_eff[df_is_eff["Layer_num"] == "19"]
fig, ax = plt.subplots(figsize=(10*2, 5*2))
sns.catplot(data=df_is_eff_conv_2d_19, x="radius_size",  kind="violin", inner="quartile", cut = cut, bw_adjust = bw_adjust)
plt.xlabel('')
plt.xticks(fontsize=16)
plt.savefig('/workspace/is_eff_conv2d_19_violin_plot_radius_size.png')

df_is_eff_conv_2d_21 = df_is_eff[df_is_eff["Layer_num"] == "21"]
fig, ax = plt.subplots(figsize=(10*2, 5*2))
sns.catplot(data=df_is_eff_conv_2d_21, x="radius_size",  kind="violin", inner="quartile", cut = cut, bw_adjust = bw_adjust)
plt.xlabel('')
plt.xticks(fontsize=16)
plt.savefig('/workspace/is_eff_conv2d_21_violin_plot_radius_size.png')

print(np.floor(np.mean(df_is_eff_conv_2d_1["radius_size"])) * 2 + 1)
print(np.floor(np.mean(df_is_eff_conv_2d_3["radius_size"])) * 2 + 1)
print(np.floor(np.mean(df_is_eff_conv_2d_5["radius_size"])) * 2 + 1)
print(np.floor(np.mean(df_is_eff_conv_2d_7["radius_size"])) * 2 + 1)
print(np.floor(np.mean(df_is_eff_conv_2d_15["radius_size"])) * 2 + 1)
print(np.floor(np.mean(df_is_eff_conv_2d_17["radius_size"])) * 2 + 1)
print(np.floor(np.mean(df_is_eff_conv_2d_19["radius_size"])) * 2 + 1)
print(np.floor(np.mean(df_is_eff_conv_2d_21["radius_size"])) * 2 + 1)

#print(np.ceil(np.mean(df_is_eff_conv_2d_21["radius_size"])) * 2 + 1)
#print(np.floor(np.mean(df_is_eff_conv_2d_21["radius_size"])) * 2 + 1)