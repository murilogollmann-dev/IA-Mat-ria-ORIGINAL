
import os
import joblib
import pandas as pd

from utils.processamento import mapear_valores
from utils.recomendacao import preparar_modelo

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'materiais_50.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__),)
MODEL_PATH = os.path.join(MODEL_DIR, 'knn_model.joblib')

def main():
    print("Carregando dados de:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df_proc = mapear_valores(df)
    modelo = preparar_modelo(df_proc, n_neighbors=5)  # treina com n_neighbors=5 para mais estabilidade
    # salvar modelo e também salvar as features usadas (opcional)
    joblib.dump({
        'modelo': modelo,
        'features': ['custo_index','densidade_kg_m3','tensile_strength_MPa','thermal_conductivity_W_mK','reciclavel','biodegradavel','melting_point_C'],
        'df_proc': df_proc  # opcional: salva df processado para inspeção
    }, MODEL_PATH)
    print("Modelo salvo em:", MODEL_PATH)

if __name__ == "__main__":
    main()
