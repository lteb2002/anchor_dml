import numpy as np
import pandas as pd
import dl_model.rere_tsne as tsne
import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import MinMaxScaler

fp = "E:\\papers\\anchor_dml\\experiments\\"
original_file = fp + "renttherunway_sample.csv"
dml_file = fp + "renttherunway_sample_anchor_dml_mlig_l3_2m.csv"
anchor_file = fp + "renttherunway_sample.csv"

img1 = fp + "img1.png"
img2 = fp + "img2.png"
img3 = fp + "img3.png"

data1 = pd.read_csv(original_file)
print(data1.head())
features1 = data1.values[:, :-1]
features1 = MinMaxScaler().fit(features1).transform(features1)
features1 = np.round(features1,6)
labels1 = data1.values[:, -1]
viz = tsne.RereTSNE(features1, labels1)
viz.save_image(img1)

data2 = pd.read_csv(dml_file)
print(data2.head())
features2 = data2.values[:, :-1]
labels2 = data2.values[:, -1]

if features2.shape[0] > 1000:
    np.random.seed(seed=123)
    ids = np.random.choice(range(features2.shape[0]), 1000, replace=False)
    features2 = features2[ids, :]
    if len(labels2) > 0:
        labels2 = labels2[ids]

tt = manifold.TSNE(n_components=2, init='pca', random_state=123)
dd2 = tt.fit_transform(features2)
tsne.RereTSNE.save_image2(dd2, labels2, img2)

data3 = pd.read_csv(anchor_file)
print(data2.head())
features3 = data3.values[:, :-1]
labels3 = data3.values[:, -1]
features3 = np.vstack([features2, features3])
labels3 = np.hstack([labels2, labels3])
dd3 = tt.fit_transform(features3)
tsne.RereTSNE.save_image2(dd3, labels3, img3)
