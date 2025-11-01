# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 19:10:45 2025

@author: aleex
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
import json
import numpy as np
from sqlalchemy import text
import itertools
import UTILS_BBDD as ub

def insert_in_batches(df, table_name, engine, batch_size=10000):
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table_name}"))
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch.to_sql(name=table_name, con=engine, if_exists='append', index=False, method='multi')
        print(f'Insertadas filas {i} a {i+len(batch)}')

def get_params():
    with open("config/params.json", "r") as f:
        params = json.load(f)
    return params


def get_medidas(): 
    medidas = pd.read_sql("""select * from fact_medida_player""", ub.get_conn("config"))
    return medidas


def get_data(conn,filtros,cond_where=""):
    config=get_params()['calculo_indicadores_nivel']
    if len(cond_where)==0:
        query = """select * from fact_ag_player_season"""
    else:
        query = """select * from fact_ag_player_season where {}""".format(cond_where)
    df = pd.read_sql(query, conn)
    df= ub.clean_df(df)
    if filtros:
        for fil in filtros:
            df= df[df[fil]>config[filtros[fil]]]
    val_cols = [i for i in df.columns if "pctscore" in i or "Performance" in i]
    if len(val_cols)>0:
        for i in val_cols:
            df.drop(i,axis=1,inplace=True)
    return df

def ajuste_normalizacion_euc(dist_matrix,aplica_sqrt=0):
    dist_matrix_no_diag = dist_matrix.copy()
    np.fill_diagonal(dist_matrix_no_diag, np.nan)

    # Calcular media y desviación estándar excluyendo diagonal
    mean_dist = np.nanmean(dist_matrix_no_diag)
    std_dist = np.nanstd(dist_matrix_no_diag)

    # Normalizamos distancias con z-score
    z_dist = (dist_matrix - mean_dist) / std_dist

    # Reescalamos z_dist a rango [0,1], invertido para similitud
    z_min = np.nanmin(z_dist)
    z_max = np.nanmax(z_dist)
    simil_matrix = 1 - (z_dist - z_min) / (z_max - z_min)
    
    # Clip para asegurar que similitud quede en [0,1]
    simil_matrix = np.clip(simil_matrix, 0, 1)
    if aplica_sqrt==1:
        # Aplica raíz cuadrada para suavizar la curva (potencia < 1)
        simil_matrix = np.sqrt(simil_matrix)
    
    # Escalamos a 0-100
    simil_matrix = simil_matrix * 100
    
    return simil_matrix

def load(df,dest,con):
    df.to_sql(name=dest, con=con, if_exists="replace", index=False)

def calcula_adecuacion(df_jugadores, df_cols, posicion):
    # Definir indicadores por grupo
    medidas = list(df_cols.medida.unique())
    # Escalamos las medidas para todos los jugadores
    scaler = MinMaxScaler()
    df_jugadores_scaled = df_jugadores.copy()
    df_jugadores_scaled[medidas] = scaler.fit_transform(df_jugadores[medidas])

    # Filtrar jugadores por posición
    df_pos = df_jugadores_scaled.copy()

    if posicion in ['GK']:
        df_jugadores['team_modelo_id']=df_jugadores['team_modelo_id'].apply(lambda x: str(x)[1:3])
    elif posicion in ['CB']:
        df_jugadores['team_modelo_id']=df_jugadores['team_modelo_id'].apply(lambda x: str(x)[:3])
    else:
        df_jugadores['team_modelo_id']=df_jugadores['team_modelo_id'].apply(lambda x: str(x))
    # Unir con perfiles escalados
    df_jug_clust = df_jugadores[['playerId', 'team_modelo_id']].drop_duplicates()
    df_jug_clust = df_jug_clust.merge(df_jugadores_scaled[['playerId'] + medidas], on='playerId', how='left')

    # Perfil medio por cluster
    perfil_medio_cluster = df_jug_clust.groupby('team_modelo_id')[medidas].mean()

    resultados = []
    perfiles_jugadores = df_pos.set_index('playerId')[medidas]

    for cluster_id, perfil_medio in perfil_medio_cluster.iterrows():
        perfil_medio_arr = perfil_medio.values.reshape(1, -1)
        perfiles_jug_arr = perfiles_jugadores.values

        # Calcular distancias euclídeas jugador vs perfil medio cluster
        distancias = euclidean_distances(perfiles_jug_arr, perfil_medio_arr).flatten()
        adecuacion_norm=ajuste_normalizacion_euc(distancias)
        # Normalizar distancias a rango [0,1], inverso para adecuacion:
        max_dist = distancias.max() if distancias.max() > 0 else 1
        adecuacion_raw = 1 - (distancias / max_dist)

        # Normalizar a 0-100 con potencia 3 para que 100 sea ultra adecuado
        adecuacion_norm = (adecuacion_raw ** 1) * 100

        df_temp = pd.DataFrame({
            'playerId': perfiles_jugadores.index,
            'team_modelo_id': cluster_id,
            'position':posicion,
            'adecuacion': adecuacion_norm
        })

        resultados.append(df_temp)

    df_result = pd.concat(resultados, ignore_index=True)
    return df_result


def calcula_adecuacion_tolerada(df_jugadores, df_cols, posicion):
    

    # 1. Seleccionar medidas
    medidas = list(df_cols.medida.unique())

    # 2. Escalar medidas
    scaler = MinMaxScaler()
    df_scaled = df_jugadores.copy()
    df_scaled[medidas] = scaler.fit_transform(df_jugadores[medidas])
    if posicion in ['GK']:
        df_scaled['team_modelo_id']=df_scaled['team_modelo_id'].apply(lambda x: str(x)[1:3])
    elif posicion in ['CB']:
        df_scaled['team_modelo_id']=df_scaled['team_modelo_id'].apply(lambda x: str(x)[:3])
    else:
        df_scaled['team_modelo_id']=df_scaled['team_modelo_id'].apply(lambda x: str(x))
    # 3. Asociar jugadores a modelos
    df_jug_clust = df_scaled[['playerId', 'team_modelo_id']].drop_duplicates()
    df_perfil = df_scaled[['playerId'] + medidas]
    df_jug_clust = df_jug_clust.merge(df_perfil, on='playerId', how='left')

    # 4. Calcular perfil medio por modelo
    perfil_medio_cluster = df_jug_clust.groupby('team_modelo_id')[medidas].mean()
    modelos_disponibles = perfil_medio_cluster.index.tolist()

    resultados = []

    # 5. Preparamos los perfiles de jugadores
    perfiles_jugadores = df_scaled.set_index(['playerId','season'])[medidas]

    for modelo_id in modelos_disponibles:
        # a) Perfil medio del modelo exacto
        perfil_objetivo = perfil_medio_cluster.loc[modelo_id].values.reshape(1, -1)
        distancias_exactas = euclidean_distances(perfiles_jugadores.values, perfil_objetivo).flatten()
        max_dist = distancias_exactas.max() if distancias_exactas.max() > 0 else 1
        adecuacion_exacta = np.round((max_dist - distancias_exactas) * 100 / max_dist, 2)

        # b) Buscar modelos tolerados (una discordancia)
        similares = [
            m for m in modelos_disponibles
            if m != modelo_id and sum(a != b for a, b in zip(str(m), str(modelo_id))) == 1
        ]

        adecuacion_tolerada = np.zeros_like(adecuacion_exacta)
        if similares:
            # Calcular adecuación para cada modelo similar y quedarnos con la mejor
            adecuaciones_similares = []

            for modelo_sim in similares:
                perfil_sim = perfil_medio_cluster.loc[modelo_sim].values.reshape(1, -1)
                distancias_sim = euclidean_distances(perfiles_jugadores.values, perfil_sim).flatten()
                max_dist_sim = distancias_sim.max() if distancias_sim.max() > 0 else 1
                adecuacion_sim = (max_dist_sim - distancias_sim) * 100 / max_dist_sim
                adecuaciones_similares.append(adecuacion_sim)

            # Tomar la media de la adecuación entre los modelos similares
            adecuacion_tolerada = np.mean(np.vstack(adecuaciones_similares), axis=0)
            adecuacion_tolerada = np.round(adecuacion_tolerada, 2)
        if len(str(modelo_id))==4:
            adecuacion_total = np.round(0.75 * adecuacion_exacta + 0.25 * adecuacion_tolerada, 2)
        else:
            adecuacion_total = adecuacion_exacta
        player_ids, seasons = zip(*perfiles_jugadores.index)
        df_temp = pd.DataFrame({
            'playerId': player_ids,
            'team_modelo_id': modelo_id,
            'season':seasons,
            'position': posicion,
            'adecuacion_exacta': adecuacion_exacta,
            'adecuacion_tolerada': adecuacion_tolerada,
            'adecuacion_total': adecuacion_total
        })

        resultados.append(df_temp)

    df_result = pd.concat(resultados, ignore_index=True)
    return df_result

def calcula_adecuacion_modelo(df_jugadores, df_cols, posicion, modelo_objetivo):
    """
    Calcula la adecuación (exacta, tolerada y total) de los jugadores a un único modelo de juego.

    Parámetros:
    - df_jugadores: DataFrame con datos de los jugadores (de cualquier modelo)
    - df_cols: DataFrame con las medidas a utilizar (columna: 'medida')
    - posicion: str (p. ej., 'GK', 'CB', 'CM', etc.)
    - modelo_objetivo: str o int que representa el modelo de juego objetivo (ej. '3233', '122', etc.)

    Retorna:
    - DataFrame con playerId, season, adecuación exacta, tolerada y total para ese modelo objetivo
    """

    # 1. Seleccionar medidas
    medidas = list(df_cols.medida.unique())

    # 2. Preparar copia + ajustar team_modelo_id
    df_scaled = df_jugadores.copy()
    if posicion == 'GK':
        df_scaled['team_modelo_id'] = df_scaled['team_modelo_id'].apply(lambda x: str(x)[1:3])
        modelo_objetivo = str(modelo_objetivo)[1:3]
    elif posicion == 'CB':
        df_scaled['team_modelo_id'] = df_scaled['team_modelo_id'].apply(lambda x: str(x)[:3])
        modelo_objetivo = str(modelo_objetivo)[:3]
    else:
        df_scaled['team_modelo_id'] = df_scaled['team_modelo_id'].apply(lambda x: str(x))
        modelo_objetivo = str(modelo_objetivo)

    # 3. Escalar medidas
    scaler = MinMaxScaler()
    df_scaled[medidas] = scaler.fit_transform(df_scaled[medidas])

    # 4. Perfil medio del modelo objetivo
    df_jug_clust = df_scaled[['playerId', 'team_modelo_id']].drop_duplicates()
    df_perfil = df_scaled[['playerId'] + medidas]
    df_jug_clust = df_jug_clust.merge(df_perfil, on='playerId', how='left')

    perfil_medio_cluster = df_jug_clust.groupby('team_modelo_id')[medidas].mean()

    if modelo_objetivo not in perfil_medio_cluster.index:
        raise ValueError(f"El modelo {modelo_objetivo} no está disponible en los datos.")

    perfil_objetivo = perfil_medio_cluster.loc[modelo_objetivo].values.reshape(1, -1)

    # 5. Perfiles de jugadores
    perfiles_jugadores = df_scaled.set_index(['playerId', 'season'])[medidas]
    distancias_exactas = euclidean_distances(perfiles_jugadores.values, perfil_objetivo).flatten()
    max_dist = distancias_exactas.max() if distancias_exactas.max() > 0 else 1
    adecuacion_exacta = np.round((max_dist - distancias_exactas) * 100 / max_dist, 2)

    # 6. Buscar modelos tolerados
    modelos_disponibles = perfil_medio_cluster.index.tolist()
    similares = [
        m for m in modelos_disponibles
        if m != modelo_objetivo and sum(a != b for a, b in zip(str(m), str(modelo_objetivo))) == 1
    ]

    adecuacion_tolerada = np.zeros_like(adecuacion_exacta)
    if similares:
        adecuaciones_similares = []
        for modelo_sim in similares:
            perfil_sim = perfil_medio_cluster.loc[modelo_sim].values.reshape(1, -1)
            distancias_sim = euclidean_distances(perfiles_jugadores.values, perfil_sim).flatten()
            max_dist_sim = distancias_sim.max() if distancias_sim.max() > 0 else 1
            adecuacion_sim = (max_dist_sim - distancias_sim) * 100 / max_dist_sim
            adecuaciones_similares.append(adecuacion_sim)
        adecuacion_tolerada = np.round(np.mean(np.vstack(adecuaciones_similares), axis=0), 2)

    # 7. Calcular adecuación total (ponderación)
    if len(str(modelo_objetivo)) == 4:
        adecuacion_total = np.round(0.75 * adecuacion_exacta + 0.25 * adecuacion_tolerada, 2)
    else:
        adecuacion_total = adecuacion_exacta

    # 8. Crear DataFrame resultado
    player_ids, seasons = zip(*perfiles_jugadores.index)
    df_resultado = pd.DataFrame({
        'playerId': player_ids,
        'season': seasons,
        'team_modelo_id': modelo_objetivo,
        'position': posicion,
        'adecuacion_exacta': adecuacion_exacta,
        'adecuacion_tolerada': adecuacion_tolerada,
        'adecuacion_total': adecuacion_total
    })

    return df_resultado

def calcula_adecuacion_tolerada_jugador(df_jugadores, df_cols, posicion, player_id_objetivo, season_objetivo):
    """
    Calcula la adecuación de un jugador concreto (y temporada) a todos los modelos disponibles.

    Parámetros:
    - df_jugadores: DataFrame completo.
    - df_cols: DataFrame con columna 'medida'.
    - posicion: string de la posición padre (ej. 'CM', 'GK', etc.).
    - player_id_objetivo: ID del jugador.
    - season_objetivo: temporada del jugador.

    Devuelve:
    - DataFrame con adecuación a cada modelo: exacta, tolerada y total.
    """

    # 1. Seleccionar medidas
    medidas = list(df_cols['medida'].unique())

    # 2. Escalar medidas
    scaler = MinMaxScaler()
    df_scaled = df_jugadores.copy()
    df_scaled[medidas] = scaler.fit_transform(df_scaled[medidas])

    # Ajustar modelo_id en función de la posición
    if posicion == 'GK':
        df_scaled['team_modelo_id'] = df_scaled['team_modelo_id'].apply(lambda x: str(x)[1:3])
    elif posicion == 'CB':
        df_scaled['team_modelo_id'] = df_scaled['team_modelo_id'].apply(lambda x: str(x)[:3])
    else:
        df_scaled['team_modelo_id'] = df_scaled['team_modelo_id'].apply(str)

    # 3. Calcular perfil medio de cada modelo
    df_jug_clust = df_scaled[['playerId', 'team_modelo_id']].drop_duplicates()
    df_perfil = df_scaled[['playerId'] + medidas]
    df_jug_clust = df_jug_clust.merge(df_perfil, on='playerId', how='left')
    perfil_medio_cluster = df_jug_clust.groupby('team_modelo_id')[medidas].mean()
    modelos_disponibles = perfil_medio_cluster.index.tolist()

    # 4. Perfil del jugador objetivo
    df_jugador = df_scaled[(df_scaled['playerId'] == player_id_objetivo) & (df_scaled['season'] == season_objetivo)]
    if df_jugador.empty:
        raise ValueError("Jugador o temporada no encontrados en el DataFrame.")
    perfil_jugador = df_jugador[medidas].values.reshape(1, -1)

    resultados = []

    for modelo_id in modelos_disponibles:
        # a) Perfil medio del modelo
        perfil_modelo = perfil_medio_cluster.loc[modelo_id].values.reshape(1, -1)
        distancia_exacta = euclidean_distances(perfil_jugador, perfil_modelo).flatten()[0]
        max_dist = distancia_exacta if distancia_exacta > 0 else 1
        adecuacion_exacta = round((max_dist - distancia_exacta) * 100 / max_dist, 2)

        # b) Buscar modelos tolerados (una discordancia)
        similares = [
            m for m in modelos_disponibles
            if m != modelo_id and sum(a != b for a, b in zip(str(m), str(modelo_id))) == 1
        ]

        adecuacion_tolerada = 0
        if similares:
            adecuaciones_similares = []
            for modelo_sim in similares:
                perfil_sim = perfil_medio_cluster.loc[modelo_sim].values.reshape(1, -1)
                dist_sim = euclidean_distances(perfil_jugador, perfil_sim).flatten()[0]
                max_dist_sim = dist_sim if dist_sim > 0 else 1
                adecuacion_sim = (max_dist_sim - dist_sim) * 100 / max_dist_sim
                adecuaciones_similares.append(adecuacion_sim)

            adecuacion_tolerada = round(np.mean(adecuaciones_similares), 2)

        # c) Ponderación total
        if len(str(modelo_id)) == 4:
            adecuacion_total = round(0.75 * adecuacion_exacta + 0.25 * adecuacion_tolerada, 2)
        else:
            adecuacion_total = adecuacion_exacta

        resultados.append({
            'playerId': player_id_objetivo,
            'season': season_objetivo,
            'team_modelo_id': modelo_id,
            'position': posicion,
            'adecuacion_exacta': adecuacion_exacta,
            'adecuacion_tolerada': adecuacion_tolerada,
            'adecuacion_total': adecuacion_total
        })

    return pd.DataFrame(resultados)

def calcula_similitud(df_jugadores_pos, df_cols, position):
    """
    df_jugadores_pos: DataFrame de jugadores de UNA posición con columnas mínimas:
        - 'playerId'
        - 'playerName'
        - 'posicion'
        - columnas de medidas de perfil (al menos las que toca según posición)

    Devuelve DataFrame con columnas:
        playerId, playerName, playerId_com, playerName_com, Similitud (0-100 normalizado)
    """

    # Definir indicadores por grupo
    medidas = list(df_cols.medida.unique())

    # Escalamos medidas
    scaler = MinMaxScaler()
    df_scaled = df_jugadores_pos.copy()
    df_scaled[medidas] = scaler.fit_transform(df_scaled[medidas])

    # Matriz perfiles escalados
    perfiles = df_scaled.set_index('playerId')[medidas].values
    playerIds = df_scaled['playerId'].values
    seasons = df_scaled['season'].values

    # Calculamos distancias euclídeas todos contra todos
    dist_matrix = euclidean_distances(perfiles)

    simil_matrix=ajuste_normalizacion_euc(dist_matrix)

    media_similitud = simil_matrix.mean()
    mask = simil_matrix > media_similitud  # matriz booleana para filtrar

    # Creamos el DataFrame final: pares (i,j) con similitud > media
    data = []
    n = len(playerIds)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # opcional: saltar comparación consigo mismo
            if not mask[i, j]:
                continue  # descartar pares con similitud baja
            data.append({
                'playerId': playerIds[i],
                'season': seasons[i],
                'playerId_com': playerIds[j],
                'season_com': seasons[j],
                'similitud': simil_matrix[i, j],
                'similitud_media_var_pct':(simil_matrix[i, j] - media_similitud) / media_similitud
            })

    df_similitud = pd.DataFrame(data)

    # Crear ID ordenado para evitar duplicados
    df_similitud['similitud_id'] = df_similitud.apply(
        lambda row: ''.join(sorted([str(row['playerId']), str(row['playerId_com'])])), axis=1
    )
    df_similitud['similitud_season'] = df_similitud.apply(
        lambda row: ''.join(sorted([str(row['season']), str(row['season_com'])])), axis=1
    )

    # Eliminar duplicados por pares y temporadas
    df_similitud = df_similitud.drop_duplicates(subset=["similitud_id", "similitud_season"], keep='first')

    return df_similitud



def calcula_similitud_jugador(df_jugadores_pos, df_cols, position, player_id_objetivo, season_objetivo):
    """
    Calcula la similitud de un jugador específico en una temporada específica frente al resto de jugadores
    de su misma posición y temporada.

    Parámetros:
    - df_jugadores_pos: DataFrame con jugadores de una posición.
    - df_cols: DataFrame con columnas 'medida'.
    - position: posición padre (ej. 'CM', 'GK', etc.).
    - player_id_objetivo: ID del jugador objetivo.
    - season_objetivo: temporada del jugador objetivo.

    Devuelve:
    - DataFrame con columnas: playerId, season, playerName, adecuacion_total, position_padre.
    """

    # Filtrar solo jugadores de la temporada objetivo
    df_filtrado = df_jugadores_pos[df_jugadores_pos['season'] == season_objetivo].copy()

    # Medidas de perfil
    medidas = list(df_cols['medida'].unique())
    
    # Escalado de medidas (entre 0 y 1)
    scaler = MinMaxScaler()
    df_filtrado[medidas] = scaler.fit_transform(df_filtrado[medidas])

    # Construcción de perfiles
    perfiles = df_filtrado.set_index('playerId')[medidas]
    player_ids = perfiles.index.tolist()

    # Comprobación de existencia del jugador objetivo
    if player_id_objetivo not in player_ids:
        raise ValueError(f"El jugador {player_id_objetivo} no está en la temporada {season_objetivo}.")

    # Diccionarios auxiliares
    seasons = df_filtrado.set_index('playerId')['season']
    player_names = df_filtrado.set_index('playerId')['playerName']

    # Perfil del jugador objetivo
    perfil_objetivo = perfiles.loc[player_id_objetivo].values.reshape(1, -1)

    # Distancias euclídeas con TODOS los jugadores (incluido él mismo)
    distancias = euclidean_distances(perfiles.values, perfil_objetivo).flatten()

    # Normalización de distancias con z-score y transformación a similitud
    mean_dist = distancias.mean()
    std_dist = distancias.std()
    z_dist = (distancias - mean_dist) / std_dist if std_dist > 0 else distancias - mean_dist
    z_min, z_max = z_dist.min(), z_dist.max()
    similitud_raw = 1 - (z_dist - z_min) / (z_max - z_min) if z_max > z_min else np.ones_like(z_dist)
    similitudes = np.clip(similitud_raw, 0, 1) * 100

    # Construir DataFrame de salida con todos los jugadores
    df_resultado = pd.DataFrame({
        'playerId': perfiles.index,
        'playerName': [player_names.loc[pid] for pid in perfiles.index],
        'season': [seasons.loc[pid] for pid in perfiles.index],
        'adecuacion_total': np.round(similitudes, 2),
        'position_padre': position
    })

    # Eliminar al jugador objetivo del resultado
    df_resultado = df_resultado[df_resultado['playerId'] != player_id_objetivo].reset_index(drop=True)

    return df_resultado

    
def main():  
    conn=ub.get_conn("config")
    df_cols=get_medidas()
    config=get_params()
    #sm.main()    
    dimpos = pd.read_sql("""select * from dim_position
                     """, conn)
    df_full=get_data(conn, config['filtros_data'])
    
    df_ade=pd.DataFrame()
    df_sim=pd.DataFrame()
    for pos in df_cols.position.unique():
        dimpos_pos = dimpos[dimpos.position_padre==pos].position_data.unique()
        cluster="cluster_{}".format(pos)
        dff=df_full[(df_full.position_padre==pos) | (df_full.position_padre2==pos)]
        df_medidas=df_cols[(df_cols.position==pos) & (df_cols.seleccion_perfil_sn==1)]
        postime=pd.DataFrame()
        for i in dimpos_pos:
            postime_aux = pd.read_sql("""select season, playerId, case when position is null then 'Sub' else position end as position, sum(minutes_played) as minutes_played_position from fact_player_stats
                                          where position = '{}' group by season, playerId, position
                                """.format(i), conn)
            postime=pd.concat([postime,postime_aux])
        postime= postime.groupby(by=["playerId","season"],as_index=False).minutes_played_position.sum()
        dff=pd.merge(dff,postime[['season','playerId','minutes_played_position']],how='left',
                          on=['season','playerId'])
        df= dff[(dff.minutes_played_position>=config["calculo_indicadores_perfil"]["minutes_threshold_pos"]*2) 
                | (dff.minutes>=config["calculo_indicadores_perfil"]["minutes_threshold"]*2)]
        df=df[df[cluster].isna()==False]
        sim=calcula_similitud(df,df_medidas,pos)
        df_sim=pd.concat([df_sim,sim])
    df_sim = df_sim.drop_duplicates(subset=["similitud_id", "similitud_season"], keep='first')
    df_sim.drop([i for i in df_sim.columns if i in ["similitud_id", "similitud_season"]],axis=1,inplace=True)
    insert_in_batches(df_sim,"mod_player_sim",conn)
        
    for pos in dimpos.position_data.unique():
        pos_padre=dimpos[dimpos.position_data==pos].position_padre.values[0]
        cluster="cluster_{}".format(pos_padre)
        df_medidas=df_cols[(df_cols.position==pos_padre) & (df_cols.seleccion_perfil_sn==1)]
        dff=df_full[(df_full.position==pos) | (df_full.position2==pos) | (df_full.position3==pos)]
        postime=pd.DataFrame()
        for i in dimpos_pos:
            postime_aux = pd.read_sql("""select season, playerId, case when position is null then 'Sub' else position end as position, sum(minutes_played) as minutes_played_position from fact_player_stats
                                          where position = '{}' group by season, playerId, position
                                """.format(i), conn)
            postime=pd.concat([postime,postime_aux])
        postime= postime.groupby(by=["playerId","season"],as_index=False).minutes_played_position.sum()
        dff=pd.merge(dff,postime[['season','playerId','minutes_played_position']],how='left',
                          on=['season','playerId'])
        
        df= dff[(dff.minutes_played_position>=config["calculo_indicadores_perfil"]["minutes_threshold_pos"]*2) 
                | (dff.minutes>=config["calculo_indicadores_perfil"]["minutes_threshold"]*2)]
        df=df[df[cluster].isna()==False]
        ade=calcula_adecuacion_tolerada(df,df_medidas,pos_padre)
        df_ade=pd.concat([df_ade,ade])
        
    load(df_ade,"mod_player_ade",conn)