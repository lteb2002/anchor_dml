import dml as dml
import pandas as pd
import numpy as np
import time
from scipy.io import arff

fp = "F:\\fruit360_exp\\"
out_path = "F:\\fruit360_exp\\"

# ns = ['spambase', 'waveform', 'HAPT', 'letter-recognition', 'magic04', 'bank_marketing', 'bank-additional',
#       'credit_card', 'BJ_DEPOSIT_RANKs',
#       'credit_risk']

# ns = ['magic04', 'bank_marketing', 'bank-additional',
#       'credit_card', 'BJ_DEPOSIT_RANKs',
#       'credit_risk']

ns = ['mind14_vec','fruit360_2']

# methods = ['itml', 'dmlmj']
# methods = ['dml_eig']
methods = ['dmlmj']


def get_dml_model(name):
    if name == 'dml_eig':
        return dml.DML_eig()
    elif name == 'lmnn':
        return dml.LMNN()
    elif name == 'itml':
        return dml.ITML()
    elif name == 'dmlmj':
        return dml.DMLMJ()
    else:
        return dml.DML_eig()


for fn in ns:
    # dn = fp + fn + '.arff'
    # data = arff.loadarff(dn)[0]
    # # print(data)
    # data = pd.DataFrame(data)
    data =pd.read_csv(fp+fn+'.csv')
    data = data.dropna()
    # print(data)
    ds = data.to_numpy()
    X = ds[:, 0:-1]
    X = X.astype(dtype=np.float32)
    # print(X.dtype)
    # y = ["label_" + str(x.decode('utf-8')) for x in ds[:, -1]]
    y = np.array(["label_" + str(x) for x in ds[:, -1]])
    for mn in methods:
        try:
            begin = time.time()
            model = get_dml_model(mn)
            # print("model:", model)
            X[np.isnan(X)] = 0
            if mn == 'lmnn' or mn == 'dml_eig':
                max_n = 1000
                np.random.seed(1)
                sample_indices = np.random.randint(0,X.shape[0],max_n)
                print(sample_indices)
                X_sample = X[sample_indices]
                # print(X)
                y_sample = y[sample_indices]
                # print(y)
            else:
                X_sample=X
                y_sample=y
            model.fit(X_sample, y_sample)
            # print(model)
            LX = model.transform(X)
            yy = np.reshape(y, (-1, 1))
            LX = np.hstack((LX, yy))
            cols = data.columns.values
            data2 = pd.DataFrame(LX, columns=cols)
            # print(cols)
            # print(data2)
            data2.to_csv(out_path + fn + '_' + mn + '.csv', header=True, index=False)
            end = time.time()
            print(mn, ' for dataset ', fn, ' time cost:', end - begin)
        except Exception as ex:
            print(ex)
