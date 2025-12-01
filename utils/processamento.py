# utils/preprocessamento.py
"""
Funções de pré-processamento robustas.
- Aceita CSV com colunas numéricas (ex.: custo_index, densidade_kg_m3, tensile_strength_MPa, thermal_conductivity_W_mK, melting_point_C, reciclavel, biodegradavel)
- Ou aceita algumas colunas categóricas comuns ('custo','peso','resistência','condutividade','temperatura_max','reciclavel','biodegradavel') e faz mapeamento.
- Garante saída sem NaNs nas features usadas pelo modelo.
"""
import pandas as pd
import numpy as np

# mapeamentos padrão caso apareçam colunas categóricas
_MAP_CUSTO = {'baixo': 0, 'baixo-médio': 1, 'médio': 2, 'médio-alto': 3, 'alto': 4}
_MAP_PESO = {'leve': 0, 'médio': 1, 'pesado': 2}
_MAP_RESIST = {'baixa': 0, 'média': 1, 'alta': 2}
_MAP_SIMNAO = {'não': 0, 'nao': 0, 'não ': 0, 'sim': 1, '1': 1, '0': 0}

def _map_col_if_exists(df: pd.DataFrame, col: str, mapa: dict):
    if col in df.columns:
        # só mapear se dtype for object ou contém strings
        if df[col].dtype == object:
            return df[col].str.strip().map(lambda v: mapa.get(v.lower(), np.nan))
        else:
            return df[col]
    else:
        return None

def mapear_valores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza/ajusta o dataframe e garante as colunas numéricas de entrada:
    -> retorna dataframe com colunas:
       ['custo_index','densidade_kg_m3','tensile_strength_MPa','thermal_conductivity_W_mK',
        'reciclavel','biodegradavel','melting_point_C','nome_material', 'id']
    """
    out = df.copy()

    # 1) colunas alternativas -> standard
    # custo_index (numérico) <- pode vir como 'custo'
    if 'custo_index' not in out.columns and 'custo' in out.columns:
        mapped = _map_col_if_exists(out, 'custo', _MAP_CUSTO)
        out['custo_index'] = mapped

    # peso -> densidade proxy
    if 'densidade_kg_m3' not in out.columns and 'peso' in out.columns:
        # se 'peso' categórico mapeia para densidade típica
        peso_col = out['peso'].astype(str).str.strip().str.lower()
        dens_map = {'leve': 1000, 'médio': 5000, 'medio': 5000, 'pesado': 8000}
        out['densidade_kg_m3'] = peso_col.map(lambda v: dens_map.get(v, np.nan))

    # resistência -> tensile_strength_MPa (tentativa de mapear)
    if 'tensile_strength_MPa' not in out.columns and 'resistencia' in out.columns:
        res = out['resistencia'].astype(str).str.strip().str.lower()
        res_map = {'baixa': 10.0, 'média': 100.0, 'media': 100.0, 'alta': 400.0}
        out['tensile_strength_MPa'] = res.map(lambda v: res_map.get(v, np.nan))

    # condutividade -> thermal_conductivity_W_mK
    if 'thermal_conductivity_W_mK' not in out.columns and 'condutividade' in out.columns:
        cond = out['condutividade'].astype(str).str.strip().str.lower()
        cond_map = {'baixa': 0.1, 'média': 1.0, 'media': 1.0, 'alta': 100.0}
        out['thermal_conductivity_W_mK'] = cond.map(lambda v: cond_map.get(v, np.nan))

    # reciclavel / biodegradavel (mapear vocabulário)
    if 'reciclavel' in out.columns:
        out['reciclavel'] = out['reciclavel'].astype(str).str.strip().str.lower().map(lambda v: _MAP_SIMNAO.get(v, np.nan))
    if 'biodegradavel' in out.columns:
        out['biodegradavel'] = out['biodegradavel'].astype(str).str.strip().str.lower().map(lambda v: _MAP_SIMNAO.get(v, np.nan))

    # melting_point_C -> temperatura máxima
    if 'melting_point_C' not in out.columns and 'temperatura_max' in out.columns:
        out['melting_point_C'] = out['temperatura_max']

    # Se algumas colunas essenciais estiverem faltando, tenta inferir ou preenche com medianas
    features = ['custo_index','densidade_kg_m3','tensile_strength_MPa','thermal_conductivity_W_mK','reciclavel','biodegradavel','melting_point_C']
    for f in features:
        if f not in out.columns:
            out[f] = np.nan

    # Converte para numérico definitivo
    out[features] = out[features].apply(pd.to_numeric, errors='coerce')

    # preencher NaNs com mediana das colunas (estratégia simples e segura)
    for f in features:
        if out[f].isna().any():
            median = out[f].median()
            # se a coluna toda é NaN, define um valor padrão razoável
            if np.isnan(median):
                # defaults razoáveis
                defaults = {
                    'custo_index': 3,
                    'densidade_kg_m3': 2000,
                    'tensile_strength_MPa': 50,
                    'thermal_conductivity_W_mK': 1.0,
                    'reciclavel': 1,
                    'biodegradavel': 0,
                    'melting_point_C': 300,
                }
                out[f] = out[f].fillna(defaults[f])
            else:
                out[f] = out[f].fillna(median)

    # manter colunas úteis
    keep = [c for c in ['id','nome_material'] if c in out.columns]
    keep += features
    return out[keep]
