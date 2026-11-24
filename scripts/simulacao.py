"""Simulação pareada e baseline fuzzy para os episódios da estufa."""

import numpy as np
import pandas as pd
from skfuzzy import trimf


ATUADORES = ['aquecedor', 'ventilacao', 'nebulizacao', 'sombreamento']
LAMBDA = 0.1
DELTA_T, DELTA_H = 1.0, 5.0


def pertinencia_triangular(x, a, b, c):
    """Triângulo com ombros permitidos quando a=b ou b=c."""
    return float(trimf(np.asarray([x]), np.asarray([a, b, c]))[0])


class ControladorFuzzy:
    """Baseline fuzzy Sugeno: 15 regras e 3 regras solares para sombreamento."""

    # 5 regras de erro térmico, 5 de erro de umidade e 5 dinâmicas = 15.
    _regras = [
        ('eT', (-9, -9, -4),  (0.00, .95, .45, .90)),
        ('eT', (-7, -3, 0),   (0.00, .65, .30, .65)),
        ('eT', (-2, 0, 2),    (.15, .15, .20, .20)),
        ('eT', (0, 3, 6),     (.65, .00, .35, .00)),
        ('eT', (3, 6, 6),     (.95, .00, .55, .00)),
        ('eH', (-25, -25, -10), (.20, .85, .00, .00)),
        ('eH', (-20, -10, 0),   (.30, .55, .10, .05)),
        ('eH', (-4, 0, 4),      (.25, .20, .25, .10)),
        ('eH', (0, 4, 10),      (.10, .05, .75, .10)),
        ('eH', (5, 10, 10),     (.00, .00, 1.00, .15)),
        ('dT', (-.7, -.7, -.2), (.80, .00, .10, .00)),
        ('dT', (-.5, -.2, 0),   (.55, .05, .15, .00)),
        ('dT', (-.15, 0, .15),  (.20, .15, .20, .05)),
        ('dT', (0, .2, .5),     (.05, .45, .15, .20)),
        ('dT', (.2, .6, .6),    (.00, .75, .10, .40)),
        ('rad', (0, 0, 180),       (0.00, 0.00, 0.00, 0.00)),
        ('rad', (80, 400, 720),    (0.00, 0.00, 0.00, 0.55)),
        ('rad', (500, 950, 1100), (0.00, 0.00, 0.00, 1.00)),
    ]

    def __call__(self, erro_t, erro_h, delta_t, radiacao=0.0):
        valores = {'eT': erro_t, 'eH': erro_h, 'dT': delta_t, 'rad': radiacao}
        por_variavel = {}
        for variavel in valores:
            regras = [(pars, out) for v, pars, out in self._regras if v == variavel]
            graus = np.array([pertinencia_triangular(valores[variavel], *pars) for pars, _ in regras])
            consequentes = np.array([out for _, out in regras])
            soma = graus.sum()
            por_variavel[variavel] = (graus @ consequentes / soma) if soma > 1e-12 else np.zeros(4)
        termica, umidade, dinamica, sol = (por_variavel[k] for k in ('eT', 'eH', 'dT', 'rad'))
        heater = np.clip(termica[0] + .15*dinamica[0] - .20*umidade[2], 0, 1)
        vent = max(termica[1], umidade[1], dinamica[1])
        neb = umidade[2]
        shade = max(termica[3], dinamica[3], sol[3])
        return np.array([heater, vent, neb, shade])


def estimar_parametros(base):
    """Estima parâmetros físicos em treino/validação, sem usar a partição teste."""
    calibracao = base[base['particao_sugerida'] != 'teste'].copy()
    if calibracao.empty:
        raise ValueError('A base não contém linhas de treino/validação')
    aq, vent, neb, somb = [calibracao[f'{nome}_pct'].to_numpy() / 100 for nome in ATUADORES]
    t, h = calibracao['temperatura_interna_c'].to_numpy(), calibracao['umidade_interna_pct'].to_numpy()
    te, he, rad = (calibracao[c].to_numpy() for c in
                   ['temperatura_externa_c', 'umidade_externa_pct', 'radiacao_solar_w_m2'])
    t1, h1 = (calibracao[c].to_numpy() for c in
              ['temperatura_interna_t1_c', 'umidade_interna_t1_pct'])
    t_sem_ruido = (t + .065*(te-t) + .0035*rad*(1-.75*somb) - .090*vent*np.maximum(t-te, 0)
                   - .020*neb + .460*aq - .018*somb*(rad/100))
    residuo_t = t1 - t_sem_ruido
    tau, sigma_h = {}, {}
    for fase, grupo in calibracao.groupby('fase_cultivo'):
        idx = grupo.index
        pos = calibracao.index.get_indexer(idx)
        h_sem_tau = (h[pos] + .060*(he[pos]-h[pos]) + 4*neb[pos]
                     - .095*vent[pos]*np.maximum(h[pos]-he[pos], 0) - .850*aq[pos]
                     + .120*somb[pos] - .020*np.maximum(t[pos]-24, 0))
        tau[fase] = float(np.median((h1[pos]-h_sem_tau)/.350))
        h_sem_ruido = h_sem_tau + .350*tau[fase]
        sigma_h[fase] = float(np.std(h1[pos]-h_sem_ruido, ddof=1))
    return {'tau': tau, 'sigma_t': float(np.std(residuo_t, ddof=1)), 'sigma_h': sigma_h}


def simular_passo(t, h, linha, acoes, fase, parametros, rng=None, adicionar_ruido=True):
    aq, vent, neb, somb = np.asarray(acoes, float)
    if not np.isfinite(acoes).all() or np.any(np.asarray(acoes) < 0) or np.any(np.asarray(acoes) > 1):
        raise ValueError('Ações devem ser quatro valores finitos em [0, 1]')
    te, he, rad = linha['temperatura_externa_c'], linha['umidade_externa_pct'], linha['radiacao_solar_w_m2']
    eps_t = rng.normal(0, parametros['sigma_t']) if adicionar_ruido else 0.0
    eps_h = rng.normal(0, parametros['sigma_h'][fase]) if adicionar_ruido else 0.0
    t1 = (t + .065*(te-t) + .0035*rad*(1-.75*somb) - .090*vent*max(t-te, 0)
          - .020*neb + .460*aq - .018*somb*(rad/100) + eps_t)
    h1 = (h + .060*(he-h) + 4*neb + .350*parametros['tau'][fase]
          - .095*vent*max(h-he, 0) - .850*aq + .120*somb
          - .020*max(t-24, 0) + eps_h)
    return float(np.clip(t1, 10, 45)), float(np.clip(h1, 20, 99))


class ControladorLookahead:
    """Refina a ação com uma predição determinística de um passo do simulador."""

    def __init__(self, controlador, parametros, peso):
        self.controlador, self.parametros, self.peso = controlador, parametros, peso

    def decidir(self, t, h, linha, delta_t):
        erro_t = linha['temperatura_referencia_c'] - t
        erro_h = linha['umidade_referencia_pct'] - h
        radiacao = linha['radiacao_solar_w_m2']
        inicial = np.asarray(self.controlador(erro_t, erro_h, delta_t, radiacao), float)
        fase = linha['fase_cultivo']
        t_prev, h_prev = simular_passo(t, h, linha, inicial, fase, self.parametros,
                                        adicionar_ruido=False)
        corrigida = np.asarray(self.controlador(
            linha['temperatura_referencia_c'] - t_prev,
            linha['umidade_referencia_pct'] - h_prev,
            t_prev - t, radiacao), float)
        return np.clip((1-self.peso)*inicial + self.peso*corrigida, 0, 1)


def simular_episodio(episodio, meta, controlador, parametros, seed):
    t, h = float(meta['temperatura_inicial_c']), float(meta['umidade_inicial_pct'])
    delta_t = 0.0
    rng = np.random.default_rng(seed)
    registros = []
    linhas = episodio.to_dict('records')
    for passo, row in enumerate(linhas):
        tref, href = row['temperatura_referencia_c'], row['umidade_referencia_pct']
        erro_t, erro_h = tref-t, href-h
        if hasattr(controlador, 'decidir'):
            acoes = controlador.decidir(t, h, row, delta_t)
        else:
            acoes = controlador(erro_t, erro_h, delta_t, row['radiacao_solar_w_m2'])
        acoes = np.clip(np.asarray(acoes, float), 0, 1)
        t1, h1 = simular_passo(t, h, row, acoes, row['fase_cultivo'], parametros, rng)
        registros.append({'episodio_id': row['episodio_id'], 'passo': row['passo_episodio'],
                          'temperatura': t, 'umidade': h, 'referencia_t': tref, 'referencia_h': href,
                          'erro_t': erro_t, 'erro_h': erro_h, **dict(zip(ATUADORES, acoes))})
        delta_t, t, h = t1-t, t1, h1
    return pd.DataFrame(registros)


def calcular_metricas(dados):
    et, eh = dados['erro_t'].to_numpy(), dados['erro_h'].to_numpy()
    return {'RMSE_T': float(np.sqrt(np.mean(et**2))),
            'RMSE_H': float(np.sqrt(np.mean(eh**2))),
            'IAE': float(np.sum(np.abs(et) + LAMBDA*np.abs(eh))),
            'violacao': float(np.mean((np.abs(et) > DELTA_T) | (np.abs(eh) > DELTA_H))),
            'consumo_E': float(dados['aquecedor'].mean()),
            'consumo_A': float(dados['nebulizacao'].mean())}


def avaliar(episodios, metadados, anfis_controller, parametros, base_seed=1000,
            fuzzy_controller=None):
    controladores = {'fuzzy': fuzzy_controller or ControladorFuzzy(), 'anfis': anfis_controller}
    linhas = []
    detalhado = {nome: {} for nome in controladores}
    for episodio_id, episodio in episodios.groupby('episodio_id', sort=True):
        meta = metadados.loc[metadados['episodio_id'] == episodio_id].iloc[0]
        for nome, controlador in controladores.items():
            sim = simular_episodio(episodio, meta, controlador, parametros, base_seed+int(episodio_id))
            detalhado[nome][int(episodio_id)] = sim
            linhas.append({'episodio_id': int(episodio_id), 'controlador': nome, **calcular_metricas(sim)})
    return pd.DataFrame(linhas), detalhado

