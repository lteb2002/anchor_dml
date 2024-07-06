from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset
import dl_model.anchor_dml.ideal_mlp as mlp
import dl_model.rere_config as cnf
import dl_model.dl_helper as hl
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, KernelPCA


class AnchorSampler:

    def __init__(self, features, labels, k=0):
        self.k = k
        self.features = features.astype(np.float32)
        self.labels = labels

    def getSamples(self, method='support_vector'):
        """
        :return:
        """
        if method == 'svm':
            # mlp_feature = self.__trainIdealMLP(self.features, self.labels)
            return self.__sampleBySVM(self.features, self.labels, self.k)
        elif method == 'clustering':
            return self.__sampleByClustering(self.features, self.k)
        elif method == 'hclustering':
            return self.__sampleByHClustering(self.features, self.k)
        elif method == 'mlig':
            return self.__sampleByMLIG(self.features)
        elif method == 'pca':
            return self.__sampleByPCA(self.features, self.k)
        elif method == 'kpca':
            return self.__sampleByKPCA(self.features, self.k)
        elif method == 'ica':
            return self.__sampleByICA(self.features, self.k)
        elif method == 'fa':
            return self.__sampleByFA(self.features, self.k)
        elif method == 'all':
            mlp_feature = self.__trainIdealMLP(self.features, self.labels)
            k1 = round(self.k / 2)
            k2 = self.k - k1
            sam1 = self.__sampleBySVM(mlp_feature, self.labels, k1)
            # sam2 = self.__sampleByICA(self.features, k2)
            sam2 = self.__sampleByClustering(self.features, k2)
            # sam3 = self.__sampleByMLIG(self.features)
            return np.vstack((sam1, sam2))

    def __trainIdealMLP(self, features, labels):
        """
        :param features:
        :param labels:
        :return:
        """
        ff = torch.from_numpy(features).to(cnf.device)
        ll = torch.from_numpy(labels).to(cnf.device)
        data_set = TensorDataset(ff, ll)
        torch.manual_seed(12)
        train_loader = torch.utils.data.DataLoader(data_set, batch_size=512, shuffle=True)
        d_in = features.shape[1]
        print("Dimension of the data:", d_in)
        d_out = len(np.unique(labels))
        print("class number:", d_out)
        print("Device:", cnf.device)
        torch.manual_seed(12)
        model = mlp.IdealMLP(d_in, d_in, d_out).to(cnf.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        for epoch in range(1, 20 + 1):
            hl.train(epoch, model, optimizer, train_loader)
        # 变换出DML数据
        ideal_data = model.transform(ff).detach().cpu().numpy()
        return ideal_data

    def __sampleBySVM(self, features, labels, k=0):
        """
        :param features:
        :param labels:
        :param k:
        :return:
        """
        clf = LinearSVC(loss="hinge", random_state=1234, max_iter=1.0E6)
        clf.fit(features, labels)
        prd = clf.predict(features)
        robust = features[labels == prd, :]
        original = self.features[labels == prd, :]
        # print("Number of robust points:", robust.shape[0])
        # obtain the support vectors through the decision function
        decision_function = clf.decision_function(robust)
        indices = np.where(np.abs(decision_function) <= 1.0)[0]
        print('Potential support vector number:', len(indices))
        sam = original
        if k == 0 or len(indices) < k:
            pass
        else:
            indices = np.argsort(np.abs(decision_function))[0:k]
            # original = original[indices,:]
            # np.random.seed(123)
            # indices = np.random.choice(len(indices),k,replace=False)
            # print(indices)
            sam = original[indices, :]
            print('Selected support vector number:', sam.shape[0])
            # print(sam.shape)
        return sam

    def __sampleByClustering(self, features, k):
        """
        :param features:
        :param k:
        :return:
        """
        centroids = KMeans(n_clusters=k).fit(features).cluster_centers_
        return centroids

    def __sampleByHClustering(self, features, k):
        """
        Get the centroids of the clusters for each label type
        :param features:
        :param k:
        :return:
        """
        ls = np.unique(self.labels)
        total = features.shape[0]
        tk = 0
        centroids = None
        for i, l in enumerate(ls):
            data = features[self.labels == l, :]
            if i + 1 != len(ls):
                sk = (data.shape[0] / total) * k
                sk = round(sk)
                tk += sk
            else:
                sk = k - tk
            # print(sk)
            cs = KMeans(n_clusters=sk).fit(data).cluster_centers_
            if centroids is None:
                centroids = cs
            else:
                centroids = np.vstack([centroids, cs])
        return centroids

    def __sampleByMLIG(self, features):
        import sympy
        threshold = 0.01
        _, inds = sympy.Matrix(features).T.rref()
        samples = features[inds, :]
        return samples

    def __sampleByPCA(self, features, k):
        (r, c) = features.shape
        # print(r,c)
        if k > min(r, c):
            k = min(r, c)
        data = features.T
        # print(k)
        transformer = PCA(n_components=k, random_state=0)
        transformer.fit(data)
        samples = transformer.transform(data).T
        # print(samples.shape)
        return samples

    def __sampleByKPCA(self, features, k):
        print('Kernel PCA')
        data = features.T
        transformer = KernelPCA(n_components=k).fit(data)
        samples = transformer.transform(data).T
        return samples

    def __sampleByICA(self, features, k):
        data = features.T
        transformer = FastICA(n_components=k).fit(data)
        samples = transformer.transform(data).T
        return samples

    def __sampleByFA(self, features, k):
        data = features.T
        transformer = FactorAnalysis(n_components=k).fit(data)
        samples = transformer.transform(data).T
        return samples
