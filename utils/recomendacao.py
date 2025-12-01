# utils/recomendacao.py
"""
Funções para treinar e recomendar.
- preparar_modelo(df): treina um NearestNeighbors com features definidas.
- recomendar_material(modelo, df_original, vetor_entrada, k=3): retorna top-k materiais.
- texto_para_vetor(texto, df): parser simples que converte descrição livre em vetor numérico.
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
import re

_FEATURES = ['custo_index','densidade_kg_m3','tensile_strength_MPa','thermal_conductivity_W_mK','reciclavel','biodegradavel','melting_point_C']

def preparar_modelo(df, n_neighbors=3):
    """
    Recebe dataframe já pré-processado (mapear_valores) e treina o NearestNeighbors.
    Retorna o modelo (fitted).
    """
    X = df[_FEATURES].values
    modelo = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    modelo.fit(X)
    return modelo

def recomendar_material(modelo, df_original, vetor_entrada, k=3):
    """
    vetor_entrada: lista/array com as mesmas features na mesma ordem _FEATURES.
    Retorna DataFrame com os top-k materiais do df_original (mantém colunas originais).
    """
    import pandas as pd
    Xq = np.array(vetor_entrada).reshape(1, -1)
    dist, idx = modelo.kneighbors(Xq, n_neighbors=k)
    resultados = df_original.iloc[idx[0]].copy()
    resultados = resultados.reset_index(drop=True)
    resultados['distancia'] = dist[0]
    return resultados

# -----------------------------
# Parser simples de texto -> vetor
# -----------------------------
def texto_para_vetor(texto: str, df_population=None):
    """
    Converte texto descritivo (português) para vetor de features.
    Estratégia:
      - procura palavras-chave (barato/médio/alto, leve/pesado, resistente, condutor, isolante,
        reciclável, biodegradável, temperatura X°C).
      - extrai números seguidos por '°C' ou 'C' para temperatura.
      - se detectar percentuais ou números isolados, tenta interpretar como densidade/força quando mencionado.
    Retorna lista na ordem _FEATURES.
    Observação: é um parser determinístico e simples — útil para interface escolar/protótipo.
    """
    t = texto.lower()

    # custo_index
    if 'barato' in t or 'baixo custo' in t:
        custo = 1
    elif 'médio' in t or 'medio' in t or 'custo médio' in t:
        custo = 3
    elif 'caro' in t or 'alto custo' in t:
        custo = 6
    else:
        custo = 3  # default médio

    # densidade proxy (palavras)
    if 'leve' in t or 'levíssimo' in t:
        dens = 900
    elif 'pesado' in t or 'pesad' in t:
        dens = 7800
    elif 'médio' in t:
        dens = 2500
    else:
        # tentar extrair número seguido de kg/m3 ou g/cm3
        match = re.search(r'(\d{2,5})\s*(kg\/m3|kg/m³)', t)
        if match:
            dens = float(match.group(1))
        else:
            match2 = re.search(r'(\d{1,4}(\.\d+)?)\s*(g\/cm3|g\/cm³)', t)
            if match2:
                dens = float(match2.group(1)) * 1000.0
            else:
                dens = df_population['densidade_kg_m3'].median() if df_population is not None else 2000

    # tensile strength proxy (resistente)
    if 'muito resistente' in t or 'alta resistência' in t:
        tens = 400.0
    elif 'resistente' in t or 'resistência alta' in t:
        tens = 200.0
    elif 'pouco resistente' in t or 'frágil' in t:
        tens = 10.0
    else:
        # tentativa de extrair MPa em texto
        m = re.search(r'(\d{2,4})\s*(mpa)', t)
        if m:
            tens = float(m.group(1))
        else:
            tens = df_population['tensile_strength_MPa'].median() if df_population is not None else 50.0

    # thermal conductivity
    if 'isolante' in t or 'baixo condutor' in t:
        cond = 0.2
    elif 'condutor' in t or 'alta condutividade' in t or 'elétrico' in t:
        cond = 100.0
    else:
        m = re.search(r'(\d{1,4}(\.\d+)?)\s*(w\/m.k|w\/m·k|w\/mk|w\/m·k)', t)
        if m:
            cond = float(m.group(1))
        else:
            cond = df_population['thermal_conductivity_W_mK'].median() if df_population is not None else 1.0

    # reciclavel / biodegradavel
    recicl = 1 if ('recicl' in t or 'reutiliz' in t) else 0
    bio = 1 if ('biodegrad' in t or 'compost' in t) else 0

    # temperatura max
    m = re.search(r'(-?\d{1,4})\s*°?\s*c', t)
    if m:
        temp = float(m.group(1))
    else:
        if 'resiste ao calor' in t or 'alta temperatura' in t:
            temp = 1000.0
        elif 'nao derrete' in t or 'não derrete' in t:
            temp = 1500.0
        else:
            temp = df_population['melting_point_C'].median() if df_population is not None else 300.0

    return [float(custo), float(dens), float(tens), float(cond), float(recicl), float(bio), float(temp)]
