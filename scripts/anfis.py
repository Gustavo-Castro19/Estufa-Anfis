"""ANFIS Takagi–Sugeno de primeira ordem, implementado com NumPy."""

from itertools import product

import numpy as np


class ANFIS:
    """ANFIS gaussiano com consequentes lineares e treinamento por gradiente."""

    def __init__(self, n_mfs=3, learning_rate=0.01, seed=42):
        self.n_mfs = n_mfs
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

    def _rules(self, x):
        z = (x[:, :, None] - self.centers[None, :, :]) / self.sigmas[None, :, :]
        log_mu = -0.5 * z**2
        log_w = np.stack([sum(log_mu[:, j, m] for j, m in enumerate(rule))
                          for rule in self.rules], axis=1)
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        return w / w.sum(axis=1, keepdims=True), z

    def _forward(self, x):
        weights, z = self._rules(x)
        basis = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        local = np.einsum('nd, rdo -> nro', basis, self.consequents)
        return np.einsum('nr,nro->no', weights, local), weights, z, basis, local

    def fit(self, x, y, epochs=100, batch_size=256, validation_data=None, patience=10):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError('x e y devem ser matrizes com o mesmo número de linhas')
        self.n_inputs, self.n_outputs = x.shape[1], y.shape[1]
        self.rules = list(product(range(self.n_mfs), repeat=self.n_inputs))
        q = np.linspace(0.05, 0.95, self.n_mfs)
        self.centers = np.column_stack([np.quantile(x[:, j], q) for j in range(self.n_inputs)])
        spread = np.maximum(np.ptp(x, axis=0), 1e-3)
        gaps = np.maximum(np.diff(self.centers, axis=0).mean(axis=0), spread / self.n_mfs)
        self.log_sigmas = np.log(np.tile(gaps, (self.n_mfs, 1)))
        self.consequents = self.rng.normal(0, 0.01, (len(self.rules), self.n_inputs + 1, self.n_outputs))
        params = [self.centers, self.log_sigmas, self.consequents]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        self.loss_history = []
        self.validation_loss_history = []
        best_loss, best_params, stale_epochs = np.inf, None, 0
        step = 0
        for epoch in range(epochs):
            order = self.rng.permutation(len(x))
            loss_sum = 0.0
            for start in range(0, len(x), batch_size):
                ix = order[start:start + batch_size]
                xb, yb = x[ix], y[ix]
                self.sigmas = np.exp(self.log_sigmas).clip(1e-3, 1e3)
                pred, weights, z, basis, local = self._forward(xb)
                err = (pred - yb) * (2 / (len(xb) * self.n_outputs))
                loss_sum += float(np.mean((pred-yb)**2)) * len(ix)
                dlogw = weights * np.einsum('no,nro->nr', err, local-pred[:, None, :])
                gc = np.einsum('nr,no,nd->rdo', weights, err, basis)
                gcenter = np.zeros_like(self.centers)
                glogsigma = np.zeros_like(self.log_sigmas)
                rule_ids = np.asarray(self.rules)
                for j in range(self.n_inputs):
                    for mf in range(self.n_mfs):
                        selected = rule_ids[:, j] == mf
                        contribution = dlogw[:, selected].sum(axis=1)
                        gcenter[j, mf] = np.sum(contribution * z[:, j, mf] / self.sigmas[j, mf])
                        glogsigma[j, mf] = np.sum(contribution * z[:, j, mf]**2)
                step += 1
                for i, grad in enumerate((gcenter, glogsigma, gc)):
                    m[i] = .9*m[i] + .1*grad
                    v[i] = .999*v[i] + .001*grad**2
                    mh = m[i] / (1-.9**step)
                    vh = v[i] / (1-.999**step)
                    params[i] -= self.learning_rate * mh / (np.sqrt(vh)+1e-8)
                self.centers[:] = np.clip(self.centers, x.min(axis=0)-2*spread, x.max(axis=0)+2*spread)
                self.log_sigmas[:] = np.clip(self.log_sigmas, np.log(1e-2), np.log(10*spread))
            self.loss_history.append(loss_sum/len(x))
            if validation_data is not None:
                vx, vy = validation_data
                self.sigmas = np.exp(self.log_sigmas)
                validation_loss = np.mean((self._forward(np.asarray(vx))[0]-vy)**2)
                self.validation_loss_history.append(float(validation_loss))
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_params = [p.copy() for p in params]
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= patience:
                        break
        if best_params is not None:
            for parameter, best in zip(params, best_params):
                parameter[:] = best
        self.sigmas = np.exp(self.log_sigmas)
        return self

    def predict(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f'Esperadas entradas com formato (n, {self.n_inputs})')
        self.sigmas = np.exp(self.log_sigmas)
        return self._forward(x)[0]

