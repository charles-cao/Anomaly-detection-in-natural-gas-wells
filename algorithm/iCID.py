import numpy as np
import os
from sklearn.metrics import roc_auc_score
import time


class IKMapper_mix():
    def __init__(self,
                 t,
                 psi,
                 ) -> None:
        self._t = t
        self._psi = psi
        self._center_list = None
        self._radius_list = None
        self.X: np.ndarray = None

    def fit(self, X: np.ndarray):
        """ Fit the model on data X.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The input instances.
        Returns
        -------
        self : object
        """
        self.X = X
        center_list = []
        for _ in range(self._t):
            center_list.append(np.random.choice(
                X.shape[0], self._psi, replace=False))
        self._center_list = np.array(center_list)

        radius_list = []
        self._embeding_metrix_inne = np.zeros((X.shape[0], self._psi*self._t))
        self._embeding_metrix_anne = np.zeros((X.shape[0], self._psi*self._t))
        for i in range(self._t):
            sample: np.ndarray = X[self._center_list[i]]

            tem1 = np.dot(np.square(X), np.ones(sample.T.shape))
            tem2 = np.dot(np.ones(X.shape), np.square(sample.T))
            p2s = tem1 + tem2 - 2 * np.dot(X, sample.T)
            s2s = p2s[self._center_list[i], :]

            row, col = np.diag_indices_from(s2s)
            s2s[row, col] = np.inf
            temp_radius_list = np.min(s2s, axis=0)
            radius_list.append(temp_radius_list)

            p2ns_index = np.argmin(p2s, axis=1)
            p2ns = p2s[range(X.shape[0]), p2ns_index]
            ind = p2ns < temp_radius_list[p2ns_index]
            self._embeding_metrix_inne[ind, (p2ns_index+i*self._psi)[ind]] = 1
            self._embeding_metrix_anne[range(
                self.X.shape[0]), (p2ns_index+i*self._psi)] = 1

        self._radius_list = np.array(radius_list)

        # self._embeding_metrix_inne = output_inne
        # self._embeding_metrix_anne = output_anne
        return self

    @property
    def embeding_mat_inne(self):
        """Get the isolation kernel map feature of fit dataset.
        """
        return self._embeding_metrix_inne

    @property
    def embeding_mat_anne(self):
        """Get the isolation kernel map feature of fit dataset.
        """
        return self._embeding_metrix_anne

    def transform_inne(self, x: np.ndarray):
        """ Compute the isolation kernel map feature of x.

        Parameters
        ----------
        x: array-like of shape (1, n_features)
            The input instances.

        Returns
        -------
        ik_value: np.array of shape (sample_size times n_members,)
            The isolation kernel map of the input instance.
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        output = np.zeros((x.shape[0], self._psi*self._t))

        for i in range(self._t):
            sample = self.X[self._center_list[i]]

            tem1 = np.dot(np.square(x), np.ones(sample.T.shape))
            tem2 = np.dot(np.ones(x.shape), np.square(sample.T))
            p2s = tem1 + tem2 - 2 * np.dot(x, sample.T)

            p2ns_index = np.argmin(p2s, axis=1)
            p2ns = p2s[range(x.shape[0]), p2ns_index]
            ind = p2ns < self._radius_list[i, p2ns_index]
            output[ind, (p2ns_index+i*self._psi)[ind]] = 1

        if x.shape[0] == 1:
            return output.reshape(-1)

        return output

    def transform_anne(self, x: np.ndarray):
        """ Compute the isolation kernel map feature of x.

        Parameters
        ----------
        x: array-like of shape (1, n_features)
            The input instances.

        Returns
        -------
        ik_value: np.array of shape (sample_size times n_members,)
            The isolation kernel map of the input instance.
        """
        # if x.shape
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        output = np.zeros((x.shape[0], self._psi*self._t))

        for i in range(self._t):
            sample = self.X[self._center_list[i]]

            tem1 = np.dot(np.square(x), np.ones(sample.T.shape))
            tem2 = np.dot(np.ones(x.shape), np.square(sample.T))
            p2s = tem1 + tem2 - 2 * np.dot(x, sample.T)

            p2ns_index = np.argmin(p2s, axis=1)
            output[range(x.shape[0]), (p2ns_index+i*self._psi)] = 1

        if x.shape[0] == 1:
            return output.reshape(-1)

        return output


def main(config):
    np.random.seed(0)
    dl = np.loadtxt(
        f'{os.path.join(config["input path"], "csv", config["input file"]+".csv")}', delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    psi = config['argument']['psi']
    t = config['argument']['t']
    window_size = config['argument']['window_size']
    scores = []
    for i in range(0, len(data), window_size):
        ik = IKMapper_mix(t=t, psi=psi)
        if i == 0:
            ik.fit(data[i:i+window_size])
        else:
            ik.fit(data[i-window_size:i+window_size])
        temp = data[i:i+window_size]

        if i == 0:
            ref_kme = np.mean(ik.transform_inne(data[i:i+window_size]), axis=0)
        else:
            ref_kme = np.mean(ik.transform_inne(data[i-window_size:i]), axis=0)

        if len(temp) == 1:
            temp_kme = ik.transform_inne(temp)
        else:
            temp_kme = np.mean(ik.transform_inne(temp), axis=0)

        dots = np.dot(ref_kme, temp_kme)
        scores += [dots for _ in range(len(temp))]
    np.savetxt(
        f'{config["output path"]}/{config["name"]}_{config["input file"]}_{roc_auc_score(1-label, scores):.8f}_{time.time():.8f}.score', scores)
