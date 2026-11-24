"""Preditor MLP de transição da estufa e busca local de ações (MPC de um passo)."""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


FASES = ('germinacao', 'vegetativo', 'floracao')
ALVO = ('temperatura_interna_t1_c', 'umidade_interna_t1_pct')
ACOES = ('aquecedor_pct', 'ventilacao_pct', 'nebulizacao_pct', 'sombreamento_pct')


def _features(frame):
    numericas = ['temperatura_interna_c', 'umidade_interna_pct',
                 'temperatura_externa_c', 'umidade_externa_pct',
                 'radiacao_solar_w_m2', 'velocidade_vento_m_s',
                 'temperatura_referencia_c', 'umidade_referencia_pct'] + list(ACOES)
    x = frame[numericas].to_numpy(float)
    fase = frame['fase_cultivo'].to_numpy()
    onehot = np.column_stack([fase == nome for nome in FASES]).astype(float)
    return np.column_stack([x, onehot])


class PreditorTransicao:
    """Compara MLPs na validação e preserva o teste para avaliação final."""

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, base, arquiteturas=((64, 32), (128, 64, 32)), max_iter=250,
            validacao=None):
        partes = {'treino': base[base['particao_sugerida'] == 'treino'],
                  'validacao': (validacao if validacao is not None else
                                base[base['particao_sugerida'] == 'validacao']),
                  'teste': base[base['particao_sugerida'] == 'teste']}
        self.x_scaler = StandardScaler().fit(_features(partes['treino']))
        self.y_scaler = StandardScaler().fit(partes['treino'][list(ALVO)].to_numpy())
        x = {k: self.x_scaler.transform(_features(v)) for k, v in partes.items()}
        y = {k: self.y_scaler.transform(v[list(ALVO)].to_numpy()) for k, v in partes.items()}
        self.candidatos, resultados = {}, []
        for arquitetura in arquiteturas:
            modelo = MLPRegressor(hidden_layer_sizes=arquitetura, activation='relu',
                                  solver='adam', alpha=1e-4, batch_size=256,
                                  learning_rate_init=1e-3, max_iter=max_iter,
                                  early_stopping=True, validation_fraction=.1,
                                  n_iter_no_change=15, random_state=self.seed)
            modelo.fit(x['treino'], y['treino'])
            self.candidatos[arquitetura] = modelo
            pred = self.y_scaler.inverse_transform(modelo.predict(x['validacao']))
            rmse = np.sqrt(np.mean((pred-partes['validacao'][list(ALVO)].to_numpy())**2, axis=0))
            resultados.append({'arquitetura': str(arquitetura), 'RMSE_T_validacao': rmse[0],
                               'RMSE_H_validacao': rmse[1],
                               'score_validacao': rmse[0] + rmse[1]/5,
                               'epocas': modelo.n_iter_})
        self.resultados_arquiteturas = resultados
        escolhida = min(resultados, key=lambda r: r['score_validacao'])['arquitetura']
        self.arquitetura = next(a for a in arquiteturas if str(a) == escolhida)
        self.modelo = self.candidatos[self.arquitetura]
        self.metricas = {}
        for nome, particao in partes.items():
            pred = self.predict_rows(particao)
            rmse = np.sqrt(np.mean((pred-particao[list(ALVO)].to_numpy())**2, axis=0))
            self.metricas[nome] = {'RMSE_T': float(rmse[0]), 'RMSE_H': float(rmse[1])}
        return self

    def predict_rows(self, rows):
        x = self.x_scaler.transform(_features(rows))
        return self.y_scaler.inverse_transform(self.modelo.predict(x))

    def predict_candidates(self, t, h, row, acoes):
        acoes = np.asarray(acoes, float)
        n = len(acoes)
        numericas = np.column_stack([
            np.full(n, t), np.full(n, h),
            np.full(n, row['temperatura_externa_c']),
            np.full(n, row['umidade_externa_pct']),
            np.full(n, row['radiacao_solar_w_m2']),
            np.full(n, row['velocidade_vento_m_s']),
            np.full(n, row['temperatura_referencia_c']),
            np.full(n, row['umidade_referencia_pct']), acoes*100])
        fase = row['fase_cultivo']
        onehot = np.tile(np.array([fase == nome for nome in FASES], float), (n, 1))
        x = self.x_scaler.transform(np.column_stack([numericas, onehot]))
        return self.y_scaler.inverse_transform(self.modelo.predict(x))


class ControladorMPC:
    """Busca local discreta de uma ação usando o MLP de transição a um passo."""

    def __init__(self, controlador_base, preditor, peso_esforco=1.0, passo=.2):
        self.base, self.preditor = controlador_base, preditor
        self.peso_esforco, self.passo = peso_esforco, passo

    def decidir(self, t, h, row, delta_t):
        erro_t = row['temperatura_referencia_c'] - t
        erro_h = row['umidade_referencia_pct'] - h
        inicial = np.asarray(self.base(erro_t, erro_h, delta_t,
                                       row['radiacao_solar_w_m2']), float)
        delta = self.passo
        alteracoes = [np.zeros(4)]
        for j in range(4):
            for sinal in (-1, 1):
                d = np.zeros(4); d[j] = sinal*delta; alteracoes.append(d)
        # Mudanças coordenadas para as duas interações físicas mais importantes.
        for a, b in ((1, 2), (0, 3)):
            for sinal in (-1, 1):
                d = np.zeros(4); d[a] = sinal*delta; d[b] = -sinal*delta
                alteracoes.append(d)
        candidatos = np.unique(np.clip(inicial + alteracoes, 0, 1), axis=0)
        previstos = self.preditor.predict_candidates(t, h, row, candidatos)
        et = (row['temperatura_referencia_c'] - previstos[:, 0]) / 1.0
        eh = (row['umidade_referencia_pct'] - previstos[:, 1]) / 5.0
        custo = (et**2 + eh**2
                 + self.peso_esforco*np.sum((candidatos-inicial)**2, axis=1)
                 + .02*(candidatos[:, 0]**2 + candidatos[:, 2]**2))
        return candidatos[int(np.argmin(custo))]

