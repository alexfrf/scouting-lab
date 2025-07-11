# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 14:42:40 2025

@author: aleex 
"""

import streamlit as st
import pandas as pd
import UTILS_BBDD as ub
import json
import PY_CALCULA_INDICADORES_PERFIL as ipe
import numpy as np
import PLAYERS_PLOTTING as pp
from PIL import Image



@st.cache_resource
def get_conn():
    con = ub.get_conn("config")
    return con
def get_params():
    with open("config/params.json", "r") as f:
        params = json.load(f)
    return params

def filtros_sidebar(dim_team,dim_competicion):
    comp_opts = list(dim_competicion[dim_competicion.pais_id.isin(["ESP","ENG","ITA","FRA","GER"])].sort_values(by=["tier_num","pais_id"]).competition.unique())
    season_opts = sorted(dim_team.season.unique())

    # Inicialización de session_state
    if "comps" not in st.session_state:
        st.session_state.comps = comp_opts[0]
    if "seasons" not in st.session_state:
        st.session_state.seasons = season_opts[-1]
    if "teams" not in st.session_state:
        # Se define más abajo con team_opts
        st.session_state.teams = None

    # Select de competición
    selected_comp = st.sidebar.selectbox("Selecciona Competición", comp_opts, index=comp_opts.index(st.session_state.comps))
    if selected_comp != st.session_state.comps:
        st.session_state.comps = selected_comp
        st.session_state.teams = None  # reinicia el equipo seleccionado
        st.rerun()


    # Select de temporada
    selected_season = st.sidebar.selectbox("Selecciona Temporada", season_opts, index=season_opts.index(st.session_state.seasons))
    if selected_season != st.session_state.seasons:
        st.session_state.seasons = selected_season
        st.session_state.teams = None  # reinicia el equipo seleccionado
        st.rerun()


    # Lista de equipos válida para esos filtros
    team_opts = sorted(dim_team[
        (dim_team.competition == st.session_state.comps) &
        (dim_team.season == st.session_state.seasons)
    ].teamName.unique())

    if st.session_state.teams not in team_opts:
        st.session_state.teams = team_opts[0] if team_opts else None
        st.rerun()


    selected_team = st.sidebar.selectbox("Selecciona Equipo", options=team_opts, index=team_opts.index(st.session_state.teams))
    if selected_team != st.session_state.teams:
        st.session_state.teams = selected_team
        st.rerun()

    return st.session_state.comps, st.session_state.seasons, st.session_state.teams
# -----------------------------
# 📡 Conexión a MySQL
# -----------------------------
@st.cache_data(ttl=86400)  # 86400 segundos = 24 horas
def get_data(filtros,cond_where=""):
    conn=get_conn()
    config=get_params()
    if len(cond_where)==0:
        query = """select * from fact_ag_player_season"""
    else:
        query = """select * from fact_ag_player_season where {}""".format(cond_where)
    df = pd.read_sql(query, conn)
    df=df.drop_duplicates()
    df= ub.clean_df(df)
    df_prot = pd.read_sql("""select * from fact_ag_player_extra""", conn)
    df_prot=df_prot.drop_duplicates()
    df_prot= ub.clean_df(df_prot)
    dim_position=pd.read_sql("""select * from dim_position""",conn)
    dim_team=pd.read_sql("""select * from dim_team""",conn)
    dim_player=pd.read_sql("""select * from dim_player where actual_sn=1""",conn)
    df_cols= pd.read_sql("""select * from fact_medida_player""",conn)
    dim_rol= pd.read_sql("""select * from dim_rol""",conn)
    dim_modelo_categoria= pd.read_sql("""select * from dim_modelo_categoria""",conn)
    dim_medida_player = pd.read_sql("""select * from dim_medida_player""",conn)
    df_time = pd.read_sql("""select fp.* from fact_player_position fp
                     """, conn)
    dim_competicion=pd.read_sql("""select * from dim_competicion fp
                     """, conn)
    dim_prototipo=pd.read_sql("""select * from dim_prototipo fp
                     """, conn)
    #dim_position=pd.read_sql("""select * from dim_position""",conn)
    #dim_position=pd.read_sql("""select * from dim_position""",conn)
    for i in df_cols.medida.unique():
        df[i]=df[i].fillna(0)
    if filtros:
        for fil in filtros:
            df= df[df[fil]>config["calculo_indicadores_nivel"]["minutes_threshold"]]
    df=pd.merge(df,dim_player[['playerId','season','age','height','weight']],
                how='left',
                on=['playerId','season'])
    df=pd.merge(df,dim_competicion[['competition','competition_desc','tier_id']],how='left',on="competition")
    return [df, df_prot, df_time, df_cols,dim_position,dim_team,dim_player,dim_rol,dim_medida_player,dim_modelo_categoria,dim_competicion,dim_prototipo]


def gestion_jugdup(data):
    dups = data[data.duplicated(subset=['playerName','season'],keep=False)]
    dups['playerName_id']=data['playerName']+"{"+data['teamName']+"}"
    data = data[~data.index.isin(list(dups.index))]
    data=pd.concat([data,dups])
    data['playerName_id'] = np.where(data.playerName_id.isna()==False,data.playerName_id, data.playerName)
    
    return data

def main():
    
    config=get_params()
    
    st.set_page_config(layout="wide", page_title="ScoutingLAB")
    st.sidebar.title("🏁 Identificación")
    st.markdown("""
    <style>
        body {
            zoom: 1;  /* 90% de escala */
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
        /* Reducir margen superior del contenedor principal */
        .block-container {
            padding-top: 3rem;
        }

        /* Ajustar margen del encabezado h1/h2/h3 según corresponda */
        h1, h2, h3 {
            margin-top: 0.1rem;
            margin-bottom: 0.1rem;
        }

        /* Reducir espacio general */
        .st-emotion-cache-1avcm0n {
            padding-top: 0.1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    image_mini=Image.open("Documentación/logo.png")
    col1,col2=st.columns([.05,.95])
    col1.image(image_mini, width=100)
    col2.markdown("## / Identificación de Jugadores", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        h3 {
            margin-top: 0.1rem;
            margin-bottom: 0.2rem;
            margin-right: 0rem;
            margin-left: 0rem;
        }
    </style>
""", unsafe_allow_html=True)
    
    
    with st.spinner("Cargando datos..."):
        
        df,df_prot,df_time,df_cols,dim_position,dim_team,dim_player,dim_rol,dim_medida_player,dim_modelo_categoria,dim_competicion,dim_prototipo = get_data(config['filtros_data'])
        df=df.drop_duplicates()
        df=gestion_jugdup(df)

    # -----------------------------
    # 🎛️ Barra lateral de filtros
    # -----------------------------
    st.sidebar.subheader("Filtros")
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] > div:first-child {
            width: 320;
        }
    </style>
    """, unsafe_allow_html=True)
    
    cols_disponibles = df.columns
    comps, seasons, teams = filtros_sidebar(dim_team,dim_competicion)
    
    # Posiciones y criterios
    posiciones_opciones = dim_position.sort_values(by="orden")['position_data'].unique().tolist()
    criterios_opciones = sorted(["Adecuación", "Similitud", "Nivel"])
    
    # Recuperar estado o usar valor por defecto
    pos_default = st.session_state.get("posiciones", posiciones_opciones[0])
    if pos_default not in posiciones_opciones:
        pos_default = posiciones_opciones[0]
    
    criterio_default = st.session_state.get("criterios", criterios_opciones[0])
    if criterio_default not in criterios_opciones:
        criterio_default = criterios_opciones[0]
    
    # Selectboxes sin claves fijas (gestión manual del estado)
    posiciones = st.sidebar.selectbox("Selecciona Posición", posiciones_opciones, index=posiciones_opciones.index(pos_default))
    
    criterios = st.sidebar.selectbox("Selecciona Criterio", criterios_opciones, index=criterios_opciones.index(criterio_default))
    
    st.session_state["posiciones"] = posiciones
    #
    st.session_state["criterios"] = criterios
    #
    
    # Filtro por temporada y equipo
    if seasons:
        df = df[df['season'] == seasons]
        if teams:
            teamid = dim_team[(dim_team.teamName == teams) & (dim_team.season == seasons)].teamId.values[0]
            modelo_juego = df[df.teamId == teamid].team_modelo_id.values[0]
            logo = dim_team[(dim_team.teamName == teams) & (dim_team.season == seasons)].img_logo.values[0]
    
    # Filtrado por posición
    if 'position' in cols_disponibles and posiciones:
        df = df[(df['position'] == posiciones) | (df['position2'] == posiciones) | (df['position3'] == posiciones)]
        position_padre = dim_position[dim_position.position_data == posiciones].position_padre.values[0]
        df = df[df[f"cluster_{position_padre}"].notna()]
        df['cluster'] = df[f"cluster_{position_padre}"]
        roles = dim_rol[dim_rol.position == position_padre]
    
        df_time = df_time.groupby(by=["playerId", "season"], as_index=False).minutes_played_position.sum()
        df = pd.merge(df, df_time[['season', 'playerId', 'minutes_played_position']], how='left', on=['season', 'playerId'])
        df = df[df[f"minutes_played_{position_padre}"] >= 300]
    
    # Indicadores
    indicadores = {
        "Adecuación": "adecuacion_total",
        "Similitud": "adecuacion_total",
        "Nivel": f"Performance_{position_padre}"
    }
    
    dfjug = pd.DataFrame()
    if criterios in ["Adecuación", "Similitud"]:
        criterio = "adecuacion_total"
        if criterios == "Similitud":
            player_sim_id = st.sidebar.selectbox("Selecciona Jugador", df[df.teamName == teams].playerName_id.unique(), index=0)
            dfjug = df[df.playerName_id == player_sim_id]
            desc = "SIMILITUD: mide el grado de parecido entre dos jugadores de la misma posición."
            accr = "SIM"
        else:
            desc = "ADECUACIÓN: mide la capacidad de un jugador de adaptarse al modelo de juego del equipo."
            accr = "ADE"
    elif criterios == "Nivel":
        criterio = f"Performance_{position_padre}"
        desc = "ADECUACIÓN: mide la capacidad de un jugador de adaptarse al modelo de juego del equipo."
        accr = "ADE"
    else:
        criterio = f"Performance_{position_padre}"
        desc = "ADECUACIÓN: mide la capacidad de un jugador de adaptarse al modelo de juego del equipo."
        accr = "ADE"
    
    # 🎚️ Otros filtros con estado persistente
    max_jugadores = df.shape[0]
    
    number_default = st.session_state.get("number", 50)
    number = st.sidebar.slider("#Jugadores en Indicadores", 10, max_jugadores, value=number_default)
    st.session_state["number"] = number
    
    if 'minutes' in cols_disponibles:
        min_minutes = int(df['minutes'].min())
        max_minutes = int(df['minutes'].max())
        selected_min_default = st.session_state.get("selected_min", 1000)
        selected_min = st.sidebar.slider("Minutos Jugados", min_minutes, max_minutes, value=selected_min_default)
        st.session_state["selected_min"] = selected_min
    
    if 'age' in cols_disponibles:
        min_age = int(df['age'].min())
        max_age = int(df['age'].max())
        selected_age_default = st.session_state.get("selected_age", max_age)
        selected_age = st.sidebar.slider("Edad Máxima", min_age, max_age, value=selected_age_default)
        st.session_state["selected_age"] = selected_age
    
    if 'height' in cols_disponibles and posiciones == "GK":
        min_height = int(df[df.height > 0]['height'].min())
        selected_hei_default = st.session_state.get("selected_hei", min_height)
        selected_hei = st.sidebar.number_input("Altura Mínima (cm)", min_value=min_height, value=selected_hei_default)
        st.session_state["selected_hei"] = selected_hei
    
    if 'competition' in cols_disponibles:
        select_league_style_default = st.session_state.get("select_league_style", "Personalizado")
        select_league_style = st.sidebar.selectbox("Selecciona Competiciones", ["Personalizado", "5 grandes ligas"], index=["Personalizado", "5 grandes ligas"].index(select_league_style_default))
        st.session_state["select_league_style"] = select_league_style
    
        all_competitions = dim_competicion.competition.unique()
        all_b5 = dim_competicion[dim_competicion.competition.isin(config['big5'])].competition.unique()
        if "5" in select_league_style:
            default_leagues = st.session_state.get("select_league", config['big5'])
            select_league = st.sidebar.multiselect("Personaliza la selección", all_b5, default=all_b5)
        else:
            default_leagues = st.session_state.get("select_league", all_competitions) 
            select_league = st.sidebar.multiselect("Personaliza la selección", all_competitions,default=all_competitions)
        if select_league:
            st.session_state["select_league"] = select_league
        else:
            st.session_state["select_league"] = comps  # fallback
                
    
   
     
    
    
    df= pd.concat([df,dfjug])
    df=df.drop_duplicates(subset=["playerId","season"],keep='first')
    
    # -----------------------------
    # 📄 Contenido principal
    # -----------------------------
    def estilo_cabeceras(df):
        return df.style.set_table_styles(
            [{
                'selector': 'th',
                'props': [('font-weight', 'bold'), 
                          ('background-color', '#f0f0f0'),
                          ('font-size', '16px')]
            }]
        )
    def calcula_tabla_adecuacion(cols,cols_aux,orden,sim=None):
        if orden!="Similitud":
            mod_player_ade = ipe.calcula_adecuacion_modelo(df, 
                                                         df_cols[df_cols.position==position_padre], 
                                                         posiciones, 
                                                         modelo_juego)
            
        else:
            mod_player_ade = ipe.calcula_similitud_jugador(df, df_cols[df_cols.position==position_padre], 
                                                           posiciones, sim, 
                                                           seasons)

        #mod_player_ade.rename({mod_player_ade.columns[-1]:"adecuacion_total"},axis=1,inplace=True)    
        cols_fun=["playerId","season"]+cols+cols_aux
    
        
        data = pd.merge(mod_player_ade[["playerId","season","{}".format("adecuacion_total")]],
                        df[[i for i in df.columns if i in cols_fun]], how="left",on=["playerId","season"])

        data = pd.merge(data,dim_team[["img_logo","teamName","season"]], 
                        how="left",on=["teamName","season"])
        data = pd.merge(data,dim_player[["logo","playerId","season","height"]], 
                        how="left",on=["playerId","season"])
        data = pd.merge(data,roles[["rol_desc","cluster"]], 
                        how="left",on=["cluster"])
        
        
        return data.sort_values(by=indicadores[orden],ascending=False)
    
    
    def calcula_tabla_similitud(sim,orden="Similitud"):
         mod_player_ade=  ipe.calcula_similitud_jugador(df, df_cols[df_cols.position==position_padre], 
                                                       position_padre, df[df.playerName_id==sim].playerId.values[0], 
                                                       seasons)

         #mod_player_ade.rename({mod_player_ade.columns[-1]:"adecuacion_total"},axis=1,inplace=True)    
         
     
         
         data = pd.merge(mod_player_ade[["playerId","playerName","season","{}".format("adecuacion_total")]],
                         df[["playerId","season","position","teamName","cluster","Performance_{}".format(position_padre),
                             "nivel_id_{}".format(position_padre)]], how="left",on=["playerId","season"])

         data = pd.merge(data,dim_team[["img_logo","teamName","season"]], 
                         how="left",on=["teamName","season"])
         data = pd.merge(data,dim_player[["logo","playerId","season"]], 
                         how="left",on=["playerId","season"])
         data = pd.merge(data,roles[["rol_desc","cluster"]], 
                         how="left",on=["cluster"])
         
         return data.sort_values(by=indicadores[orden],ascending=False)   

        
    
    col_logo, col_title = st.columns([0.05, 0.95])

    with col_logo:
        st.image(logo, width=60)  # Ajusta el tamaño del logo
    posiciones_nombre = dim_position[dim_position.position_data==posiciones].position_desc.values[0]
    jug_sim=""
    #colti1, colti2= st.columns([0.7, 0.3])
    with col_title:
        if criterios=="Nivel":
            st.subheader(f"| Top {posiciones_nombre} por {criterios.upper()}")
        elif criterios=="Adecuación":
            st.subheader(f"| Top {posiciones_nombre} por {criterios.upper()} a Modelo de Juego del Equipo")
        else:
            jug_sim= df[df.playerName_id==player_sim_id].playerName.values[0]
            team_sim= df[df.playerName_id==player_sim_id].teamName.values[0]
            st.subheader(f"| Top {posiciones_nombre} por {criterios.upper()} a {jug_sim} ({team_sim})")
    #colti2.markdown("  Dispersión de {} por Rol de {}".format(criterio,position_padre))
    #○tabla_adecuacion['playerName'] = tabla_adecuacion['playerName'].apply(lambda x: f'<span style="font-size:18px;">{x}</span>')
    

    df["nivel_id_{}".format(position_padre)]=df["nivel_id_{}".format(position_padre)].apply(lambda x: x.replace("Nivel","").replace("All","").replace(position_padre,"").replace("_","").strip())
    columnas_tabla = ["logo","playerName","img_logo","teamName","tier_id","position","age","playerName_id",
                             "rol_desc","competition",
                             "minutes",
                             "nivel_id_{}".format(position_padre),
                             "Performance_{}".format(position_padre),
                             "adecuacion_total"]
    columnas_aux=["cluster",'Performance_{}_ATAQUE'.format(position_padre)]
    if criterios=="Similitud":
        tabla_adecuacion=calcula_tabla_adecuacion(columnas_tabla,columnas_aux,criterios, df[df.playerName_id==player_sim_id].playerId.values[0])
    else:
        tabla_adecuacion=calcula_tabla_adecuacion(columnas_tabla,columnas_aux,criterios)
        
    df_own = tabla_adecuacion[(tabla_adecuacion.teamName==teams) & (tabla_adecuacion[criterio].isna()==False)].sort_values(by="minutes",ascending=False).head(4)
    df_own= df_own.sort_values(by=criterio,ascending=False)
    tabla_adecuacion = tabla_adecuacion[tabla_adecuacion['competition'].isin(select_league)]
    if posiciones:
            tabla_adecuacion = tabla_adecuacion[tabla_adecuacion['minutes']>=selected_min]
    if posiciones:
                tabla_adecuacion = tabla_adecuacion[tabla_adecuacion['age']<=selected_age]
    if posiciones=="GK":
                tabla_adecuacion = tabla_adecuacion[(tabla_adecuacion['height']>=selected_hei) | 
                                                    (tabla_adecuacion.height==0) | 
                                                    (tabla_adecuacion.height.isna()==True)]
    df = pd.merge(df,roles[["rol_desc","cluster"]], 
                    how="left",on=["cluster"]
                    )
    tabla_adecuacion=pd.concat([tabla_adecuacion,df[df.teamName==teams]])
    tabla_adecuacion = tabla_adecuacion.drop_duplicates(subset=['playerId','season'],keep='first')
    div1,div2,div2b,div3a,div3b = st.columns([0.3,.4,.1,.3,0.2])
    
     
    col1, col2= st.columns([0.65,0.35]) 
    columnas_tabla= [i for i in columnas_tabla if "competition" not in i and "playerName_id" not in i]
    ts=col1.dataframe(tabla_adecuacion[columnas_tabla], 
                   use_container_width=True, hide_index=True,
                   column_config={
                       "playerName":"Jugador",
                       "teamName":"Equipo",
                       "tier_id":"Liga",
                       "position":"Pos.",
                       "minutes":"Min.",
                       "age":"Edad",
                       "rol_desc":"Rol",
                       "nivel_id_{}".format(position_padre):"Nivel",
                       "img_logo": st.column_config.ImageColumn(""
                   ),
                       "logo": st.column_config.ImageColumn(
                           ""
                       ),
                       "adecuacion_total":st.column_config.ProgressColumn(
                               "{}".format(accr),min_value=20,max_value=99,width="small",
                               help="{}".format(desc),
                               format="%f"),
                                   "Performance_{}".format(position_padre):st.column_config.ProgressColumn(
                                           "PER",min_value=20,max_value=99,width="small",
                                           help="PERFORMANCE: Mide el nivel neto del jugador",
                                           format="%f")
                       },
                   on_select="rerun")
    df=pd.merge(df,tabla_adecuacion[["playerId","season"]+[i for i in tabla_adecuacion.columns if i not in df.columns]],
                how='left',on=["playerId","season"])
    filtered_df = tabla_adecuacion.iloc[ts.selection.rows]
    if filtered_df.shape[0]==0:
        tss=tabla_adecuacion.head(number)
    else:
        tss=filtered_df
    
    if filtered_df.shape[0]==0:
        col2.plotly_chart(pp.boxplot_xaxisv1_plotly(tabla_adecuacion,
            tss[columnas_tabla].sort_values(by=criterio,ascending=False).index.values[0],
            criterio,
            "rol_desc",
            teams,posiciones, criterios, position_padre
            ))
    else:
        col2.plotly_chart(pp.boxplot_xaxisv1_plotly(tabla_adecuacion,
            filtered_df[columnas_tabla].sort_values(by=criterio,ascending=False).index.values[0],
            criterio,
            "rol_desc",
            teams,posiciones, criterios, position_padre
            ))
    try:
        a, b, c= div1.columns([.15,.4,.4])
        if filtered_df.shape[0]>0:
            b.image(dim_player[dim_player.playerName==tss.playerName.values[0]].logo.values[0],width=80,
                    caption=f"{dim_player[dim_player.playerName==tss.playerName.values[0]].playerName.values[0]}")
            a.image(dim_team[dim_team.teamName==tss.teamName.values[0]].img_logo.values[0], width=30)
        if criterios=="Adecuación" or criterios=="Nivel":    
            c.metric("**{}**".format("ADE"),"{:.2f}".format(tss["adecuacion_total"].mean()),"{:.0%}".format((tss["adecuacion_total"].mean()-tabla_adecuacion["adecuacion_total"].mean())/tabla_adecuacion["adecuacion_total"].mean()),border=True,
                     help="ADECUACIÓN: mide el encaje potencial de un jugador en un modelo de juego. Comparación respecto a la media de la posición. BASE: Adecuación del jugador seleccionado (si existe). Sino: media del top {} por {}".format(number,criterios))
        else:
            c.metric("**{}**".format("SIM"),"{:.2f}".format(tss["adecuacion_total"].mean()),"{:.0%}".format((tss["adecuacion_total"].mean()-tabla_adecuacion["adecuacion_total"].mean())/tabla_adecuacion["adecuacion_total"].mean()),border=True,
                     help="SIMILITUD: mide el grado de parecido entre dos jugadores. Comparación respecto a la media de la posición. BASE: Adecuación del jugador seleccionado (si existe). Sino: media del top {} por {}".format(number,criterios))
    except:
        pass
    df_prot_comp=df_prot[(df_prot.competition==comps) & (df_prot.position==position_padre)]
    media_per=df_prot[(df_prot.competition==comps) & (df_prot.position==position_padre) & (df_prot.prototipo_id=="Promedio_All")].Performance.mean()
    media_def=df_prot[(df_prot.competition==comps) & (df_prot.position==position_padre) & (df_prot.prototipo_id=="Promedio_All")].Performance_DEFENSA.mean()
    media_con =df_prot[(df_prot.competition==comps) & (df_prot.position==position_padre) & (df_prot.prototipo_id=="Promedio_All")].Performance_CONSTRUCCION.mean()
    media_ata=df_prot[(df_prot.competition==comps) & (df_prot.position==position_padre) & (df_prot.prototipo_id=="Promedio_All")].Performance_ATAQUE.mean()
    media_par=df_prot[(df_prot.competition==comps) & (df_prot.position==position_padre) & (df_prot.prototipo_id=="Promedio_All")].Performance_PARADAS.mean()
    d,a,b,c=div2.columns(4)
    d.metric("**PER**", "{:.0f}".format(df[df.playerId.isin(tss.playerId.unique())]["Performance_{}".format(position_padre)].mean()), "{:.0%}".format((df[df.playerId.isin(tss.playerId.unique())]["Performance_{}".format(position_padre)].mean() - media_per) / media_per),border=True,
             help="Mide el rendimiento o nivel neto de un jugador. Comparación respecto a la media de la posición. BASE: Adecuación del jugador seleccionado (si existe). Sino: media del top {} por {}".format(number,criterios))
    a.metric("**DEF**", "{:.0f}".format(df[df.playerId.isin(tss.playerId.unique())]["Performance_DEFENSA_{}".format(position_padre)].mean()), "{:.0%}".format((df[df.playerId.isin(tss.playerId.unique())]["Performance_DEFENSA_{}".format(position_padre)].mean() - media_def) / media_def),border=True,
             help="Mide el rendimiento o nivel neto de un jugador a efectos de DEFENSA -capacidad para evitar tiros o recuperar el balón-. Comparación respecto a la media de la posición. BASE: Adecuación del jugador seleccionado (si existe). Sino: media del top {} por {}".format(number,criterios))
    b.metric("**CON**", "{:.0f}".format(df[df.playerId.isin(tss.playerId.unique())]["Performance_CONSTRUCCION_{}".format(position_padre)].mean()), "{:.0%}".format((df[df.playerId.isin(tss.playerId.unique())]["Performance_CONSTRUCCION_{}".format(position_padre)].mean() - media_con) / media_con),border=True,
             help="Mide el rendimiento o nivel neto de un jugador a efectos de CONSTRUCCIÓN -capacidad para mejorar o prolongar las posesiones del equipo-. Comparación respecto a la media de la posición. BASE: Adecuación del jugador seleccionado (si existe). Sino: media del top {} por {}".format(number,criterios))
    try:
        c.metric("**ATA**", "{:.0f}".format(df[df.playerId.isin(tss.playerId.unique())]["Performance_ATAQUE_{}".format(position_padre)].mean()), "{:.0%}".format((df[df.playerId.isin(tss.playerId.unique())]["Performance_ATAQUE_{}".format(position_padre)].mean() - media_ata) / media_ata),border=True,
                 help="Mide el rendimiento o nivel neto de un jugador a efectos de ATAQUE -capacidad para generar peligro o convertirlo en gol-. Comparación respecto a la media de la posición. BASE: Adecuación del jugador seleccionado (si existe). Sino: media del top {} por {}".format(number,criterios))
    except:
        pass
    try:
       c.metric("**PAR**", "{:.0f}".format(df[df.playerId.isin(tss.playerId.unique())]["Performance_PARADAS_{}".format(position_padre)].mean()), "{:.0%}".format((df[df.playerId.isin(tss.playerId.unique())]["Performance_PARADAS_{}".format(position_padre)].mean() - media_par) / media_par),border=True,
                help="Mide el rendimiento o nivel neto de un jugador a efectos de PARADAS -capacidad para evitar goles en contra-. Comparación respecto a la media de la posición. BASE: Adecuación del jugador seleccionado (si existe). Sino: media del top {} por {}".format(number,criterios))
    except:
        pass
    #if filtered_df.shape[0]>0:
    #    div2b.metric("**POS SEC**",df[df.playerId.isin(tss.playerId.unique())].position2.values[0],df[df.playerId.isin(tss.playerId.unique())].position3.values[0],
    #             border=True,delta_color="off")
    #if jug_sim:
    #    div3.markdown(
    #            f"""
    #            <br><br><br>
    #                            **{posiciones} - {criterios} por Rol | {jug_sim.upper()}**
    #                            
    #            """,
      #          unsafe_allow_html=True
    #            )
    #else:
    #    div3.markdown(
    #            f"""
    #            <br><br><br>
    #                            **{posiciones} - {criterios} por Rol**
    #                            
    #            """,
    #              unsafe_allow_html=True
    #            )
    
    #st.markdown(people)
    
    #df_own = tabla_adecuacion[(tabla_adecuacion.teamName==teams) & (tabla_adecuacion[criterio].isna()==False)].sort_values(by="minutes",ascending=False).head(4)
    #df_own= df_own.sort_values(by=criterio,ascending=False)
    st.caption("{} jugadores devueltos para la posición de {}".format(tabla_adecuacion.shape[0],posiciones_nombre))
    
    kpi1,_,kpi2,_,kpi3,_,kpi4,_,_,kpi5 = st.columns([.22,.03,.22,.03,.22,.03,.22,.01,.01,.5])
    try:
        a, b, c= kpi1.columns([.15,.4,.4])
        b.image(dim_player[dim_player.playerName==df_own.playerName.values[0]].logo.values[0],width=70,
                caption=f"{dim_player[dim_player.playerName==df_own.playerName.values[0]].playerName.values[0]}")
        a.image(logo, width=80)
    
        c.metric(criterio[:3].upper(),"{:.0f}".format(tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[0]][criterio].values[0]),"{:.0%}".format((tss[criterio].mean() - tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[0]][criterio].values[0])/ tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[0]][criterio].values[0]),
                 border=False)
    except:
        pass
    try:
        a, b, c= kpi2.columns([.15,.4,.4])
        b.image(dim_player[dim_player.playerName==df_own.playerName.values[1]].logo.values[0],width=70,
                caption=f"{dim_player[dim_player.playerName==df_own.playerName.values[1]].playerName.values[0]}")
        a.image(logo, width=80)
        
        c.metric(criterio[:3].upper(),"{:.0f}".format(tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[1]][criterio].values[0]),"{:.0%}".format((tss[criterio].mean() - tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[1]][criterio].values[0])/ tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[1]][criterio].values[0]),
                 border=False)
    except:
        pass
    
    try:

        a, b, c= kpi3.columns([.15,.4,.4])
        b.image(dim_player[dim_player.playerName==df_own.playerName.values[2]].logo.values[0],width=70,
                caption=f"{dim_player[dim_player.playerName==df_own.playerName.values[2]].playerName.values[0]}")
        a.image(logo,width=80)
       
        c.metric(criterio[:3].upper(),"{:.0f}".format(tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[2]][criterio].values[0]),"{:.0%}".format((tss[criterio].mean() - tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[2]][criterio].values[0])/ tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[2]][criterio].values[0]))
    except:
        pass
    
    try:

        a,b, c= kpi4.columns([.15,.4,.4])
        b.image(dim_player[dim_player.playerName==df_own.playerName.values[3]].logo.values[0],width=70,
                caption=f"{dim_player[dim_player.playerName==df_own.playerName.values[3]].playerName.values[0]}")
        a.image(logo,width=80)
        c.metric(criterio[:3].upper(),"{:.0f}".format(tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[3]][criterio].values[0]),"{:.0%}".format((tss[criterio].mean() - tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[3]][criterio].values[0])/ tabla_adecuacion[tabla_adecuacion.playerName_id==df_own.playerName_id.values[3]][criterio].values[0]))
    except:
        pass
    
    if criterios!="Nivel":
        
        kpi5.dataframe(dim_rol[dim_rol.position==position_padre][['rol_desc','rol_desc_long']],
                       height=100,hide_index=True,use_container_width=True,
                       column_config={
                           "rol_desc":"Rol","rol_desc_long":"Descripción"
                           })
    else:
        medias_prot = df_prot_comp.groupby(by="prototipo_id",as_index=False).Performance.mean()
        medias_prot = pd.merge(medias_prot,dim_prototipo,how='inner',left_on="prototipo_id",right_on="nivel_id")
        kpi5.dataframe(medias_prot.sort_values(by='orden')[['nivel_desc','Performance','nivel_desc_long']],
                       height=120,hide_index=True,use_container_width=True,
                       column_config={
                           "nivel_desc":"Nivel","nivel_desc_long":"Descripción",'Performance':'PER Medio Comp.'
                           })
    kpi5.caption("Doble click sobre celda para ver texto completo")
    
    div3a.metric("**Mejor Rol - {}**".format(criterios.replace(f"_{position_padre}","").upper()),"{}".format(tabla_adecuacion.groupby(by="rol_desc")[criterio].mean().sort_values(ascending=False).index.values[0]))
    div3b.metric("**Mejor Jugador - Rol Top**".format(criterios.replace(f"_{position_padre}","").upper()),"{}".format(tabla_adecuacion[tabla_adecuacion.rol_desc==tabla_adecuacion.groupby(by="rol_desc")[criterio].mean().sort_values(ascending=False).index.values[0]].sort_values(by=criterio,ascending=False).playerName.values[0]))
    
    
    st.divider()
    col4, col5= st.columns([1,1]) 
    col4.markdown("""### :black_circle: Dashboard por Jugador""")
    
    col6sel,col8sel,col7sel= st.columns([.6,.15,.25]) 
    
    tabla_adecuacion['sel']=np.where(tabla_adecuacion.playerId.isin(tss.playerId.unique()),1,0)

    select_pl = col6sel.selectbox(
            "Selecciona Jugador:",
            tuple(tabla_adecuacion.sort_values(by=['sel',criterio],ascending=False)[tabla_adecuacion.playerId.isin(tabla_adecuacion.playerId.unique())].playerName_id.unique())
            )
    select_col_sc = col8sel.selectbox(
            "Selecciona Indicador:",
            tuple([i for i in indicadores]+list(df_cols[(df_cols.position==position_padre) & (df_cols.calculo_sn==1)].fancy_name_esp.unique()))
            )
    
    list_comparar = ["Competiciones Seleccionadas",
                     "Competición del Jugador - {}".format(dim_team[dim_team.teamName==tabla_adecuacion[tabla_adecuacion.playerName_id==select_pl].teamName.values[0]].competition.values[0]),
                     "Competición del Equipo - {}".format(dim_team[dim_team.teamName==teams].competition.values[0])]
    
    ligas_comp = col7sel.selectbox("Selecciona Competición a Comparar", list_comparar)
    
    col6, col7, col8= st.columns([1,.15,.85]) 
    if ligas_comp==list_comparar[:1]:
        col6.pyplot(pp.sradar(select_pl,df,df_prot_comp,"Promedio_All", "Nivel Top_All",
                           list(df_cols[(df_cols.position==position_padre) & (df_cols.calculo_sn==1)].medida.unique()),
                           position_padre,
                           dim_player,
                           dim_team,
                           seasons,
                           list(df_cols[(df_cols.position==position_padre) & (df_cols.calculo_sn==1)].fancy_name_esp.unique())
                           
            )
                    )
    else:
        col6.pyplot(pp.sradar(select_pl,df,df_prot[(df_prot.competition==dim_team[dim_team.teamName==teams].competition.values[0]) & (df_prot.position==position_padre)],
                           "Promedio_All", "Nivel Top_All",
                           list(df_cols[(df_cols.position==position_padre) & (df_cols.calculo_sn==1)].medida.unique()),
                           position_padre,
                           dim_player,
                           dim_team,
                           seasons,
                           list(df_cols[(df_cols.position==position_padre) & (df_cols.calculo_sn==1)].fancy_name_esp.unique())
            )
    
                    )
    col7.pyplot(pp.pitch_maker(select_pl,df,roles[roles.cluster==df[(df.playerName_id==select_pl) & (df.season==seasons)]["cluster_{}".format(position_padre)].values[0]].rol_desc.values[0],seasons,'purple'))
    if select_col_sc not in indicadores:
        if ligas_comp==list_comparar[1]:
            col8.plotly_chart(pp.scatter_xaxisv1_plotly(df[(df.competition==dim_team[dim_team.teamName==tabla_adecuacion[tabla_adecuacion.playerName==select_pl].teamName.values[0]].competition.values[0]) | (df.playerName==select_pl)], tabla_adecuacion[tabla_adecuacion.playerName==select_pl].index[0],
                                                     df_cols[df_cols.fancy_name_esp==select_col_sc].medida.values[0],
            "rol_desc",position_padre
                        ))
        elif ligas_comp==list_comparar[-1]:
            col8.plotly_chart(pp.scatter_xaxisv1_plotly(df[(df.competition==dim_team[dim_team.teamName==teams].competition.values[0]) | (df.playerName==select_pl)], tabla_adecuacion[tabla_adecuacion.playerName==select_pl].index[0],
                                                     df_cols[df_cols.fancy_name_esp==select_col_sc].medida.values[0],
            "rol_desc",position_padre
                        ))
        else:
            
            col8.plotly_chart(pp.scatter_xaxisv1_plotly(df, tabla_adecuacion[tabla_adecuacion.playerName_id==select_pl].index[0],
                                                     df_cols[df_cols.fancy_name_esp==select_col_sc].medida.values[0],
            "rol_desc",position_padre
                        ))
        
        
        
    else:
        
        if ligas_comp==list_comparar[1]:
            col8.plotly_chart(pp.scatter_xaxisv1_plotly(df[(df.competition==dim_team[dim_team.teamName==tabla_adecuacion[tabla_adecuacion.playerName==select_pl].teamName.values[0]].competition.values[0]) | (df.playerName==select_pl)], tabla_adecuacion[tabla_adecuacion.playerName==select_pl].index[0],
                                                     indicadores[select_col_sc],
            "rol_desc",position_padre
                        ))
        elif ligas_comp==list_comparar[-1]:
            col8.plotly_chart(pp.scatter_xaxisv1_plotly(df[(df.competition==dim_team[dim_team.teamName==teams].competition.values[0]) | (df.playerName_id==select_pl)], tabla_adecuacion[tabla_adecuacion.playerName==select_pl].index[0],
                                                     indicadores[select_col_sc],
            "rol_desc",position_padre
                        ))
        else:
            
            col8.plotly_chart(pp.scatter_xaxisv1_plotly(df, tabla_adecuacion[tabla_adecuacion.playerName_id==select_pl].index[0],
                                                     indicadores[select_col_sc],
            "rol_desc",position_padre
                        ))
    
    
    if ligas_comp==list_comparar[1]:
        col8.pyplot(pp.plot_percentiles(select_pl, df[(df.competition==dim_team[dim_team.teamName==tabla_adecuacion[tabla_adecuacion.playerName_id==select_pl].teamName.values[0]].competition.values[0]) | (df.playerName==select_pl)], 
                            df_cols[(df_cols.position==position_padre) & (df_cols.calculo_sn==1)], 
                            seasons,
                            "Performance_{}".format(position_padre),
                            posiciones
                    )
                    )
    elif ligas_comp==list_comparar[-1]:
        col8.pyplot(pp.plot_percentiles(select_pl, df[(df.competition==dim_team[dim_team.teamName==teams].competition.values[0]) | (df.playerName_id==select_pl)], 
                            df_cols[(df_cols.position==position_padre) & (df_cols.calculo_sn==1)], 
                            seasons,
                            "Performance_{}".format(position_padre),
                            posiciones
                    )
                    )
    else:
        
        col8.pyplot(pp.plot_percentiles(select_pl, df, 
                        df_cols[(df_cols.position==position_padre) & (df_cols.calculo_sn==1)], 
                        seasons,
                        "Performance_{}".format(position_padre),
                        posiciones
                )
                )
    
    col4, col5= st.columns([1,1]) 
    col4.markdown("""### :black_circle: Dashboard Comparativo""")
    col4b, col5= st.columns([1,1]) 
    option_viz = st.radio(
                "Selecciona Visualización Comparativa",["***Radar vs. Jugador***","***Mapa de Calor vs. Prototipos***"],horizontal =True)
    col8,col9,col10 = st.columns([0.6,0.6,.8]) 

    df = pd.concat([df[df.playerName_id==select_pl],df])
    df = df.drop_duplicates(subset='playerId',keep='first')
    
    select_pl1 = col8.selectbox(
            "Mostrar comparación de:",
            tuple(df.playerName_id.unique()))

   
    if "Radar" in option_viz:
        select_pl2 = col9.selectbox(
                "y:",
                tuple(pd.concat([df[df.teamName==teams],df]).drop_duplicates(subset='playerId',keep='first').playerName_id.unique()
                      )
                )
    col6, col7= st.columns([1.2,.8])
    df_cols_sel= df_cols[df_cols.position==position_padre]
    df_cols_sel =pd.merge(df_cols_sel[df_cols_sel.calculo_sn==1],dim_medida_player[["position","medida","acronimo"]],on=["position","medida"],how='left')
    
    if "Radar" in option_viz:
        col6.pyplot(pp.sradar_comp(select_pl1,select_pl2,df,df_prot_comp,"Promedio_All", "Nivel Top_All",
                       list(df_cols[(df_cols.position==position_padre) & (df_cols.calculo_sn==1)].medida.unique()), 
                       position_padre, dim_player,dim_team,
                       seasons,seasons,
                  nombres=list(df_cols[(df_cols.position==position_padre) & (df_cols.calculo_sn==1)].fancy_name_esp.unique())
        )
                    )
    else:
        ligas_comp=col9.selectbox("Selecciona competición a comparar",list_comparar[1:])
        prototipos= config['seleccion_prototipos']
        
        if ligas_comp==list_comparar[1]:
            df_prot_comp=df_prot[(df_prot.competition==df[df.playerName_id==select_pl1].competition.values[0]) & (df_prot.position==position_padre)]
        df_prot_comp.rename({"Performance":"Performance_{}".format(position_padre)},axis=1,inplace=True)
        col6.pyplot(pp.mapa_calor_prototipos(df,df_prot_comp[(df_prot_comp.position==position_padre)], prototipos,
                              list(df_cols_sel[df_cols_sel.calculo_sn==1].medida.unique())+["Performance_{}".format(position_padre)],
                              list(df_cols_sel[df_cols_sel.calculo_sn==1]["fancy_name_esp"].unique())+["PER"],
                              select_pl1,seasons, True)
                  )
    
    col7.dataframe(df_cols_sel[["fancy_name_esp","categoria_desc"]],
                       use_container_width=True, hide_index=True,height=130,
                       column_config={
                           "fancy_name_esp":"Medida",
                           "categoria_desc":"Categoría"
                           }
                       )
    
    
    columnas_tabla=["logo","playerName","img_logo","rol_desc","nivel_id_{}".format(position_padre),"adecuacion_total",
                    "Performance_{}".format(position_padre)]
    tabla_sim1 = calcula_tabla_similitud(select_pl1)
    col7.markdown("**Similitud de {} vs. Jugadores de {}**".format(select_pl1,teams))
    col7.dataframe(tabla_sim1[tabla_sim1.teamName==teams].head(10)[columnas_tabla], 
                   use_container_width=True, hide_index=True,
                   column_config={
                       "playerName":"Jugador",
                       "position":"Posición",
                       "rol_desc":"Rol",
                       "nivel_id_{}".format(position_padre):"Nivel",
                       "img_logo": st.column_config.ImageColumn(""
                   ),
                       "logo": st.column_config.ImageColumn(
                           ""
                       ),
                       "adecuacion_total":st.column_config.ProgressColumn(
                               "{}".format("Similitud"),min_value=20,max_value=99,width="small",
                               help="SIMILITUD:Mide el grado de distancia entre dos perfiles de jugador.",
                               format="%f"),
                                   "Performance_{}".format(position_padre):st.column_config.ProgressColumn(
                                           "PER",min_value=20,max_value=99,width="small",
                                           help="PERFORMANCE: Mide el nivel neto del jugador",
                                           format="%f")
                       })
    col7.markdown("**Jugadores Similares a {}**".format(select_pl1))
    col7.dataframe(tabla_sim1.head(50)[columnas_tabla], 
                   use_container_width=True, hide_index=True,
                   column_config={
                       "playerName":"Jugador",
                       "position":"Posición",
                       "rol_desc":"Rol",
                       "nivel_id_{}".format(position_padre):"Nivel",
                       "img_logo": st.column_config.ImageColumn(""
                   ),
                       "logo": st.column_config.ImageColumn(
                           ""
                       ),
                       "adecuacion_total":st.column_config.ProgressColumn(
                               "{}".format("Similitud"),min_value=20,max_value=99,width="small",
                               help="SIMILITUD:Mide el grado de distancia entre dos perfiles de jugador.",
                               format="%f"),
                                   "Performance_{}".format(position_padre):st.column_config.ProgressColumn(
                                           "PER",min_value=20,max_value=99,width="small",
                                           help="PERFORMANCE: Mide el nivel neto del jugador",
                                           format="%f")
                       },height=300)
    
    if "Radar" in option_viz:
        tabla_sim2 = calcula_tabla_similitud(select_pl2)
        col7.markdown("**Jugadores Similares a {}**".format(select_pl2))
        col7.dataframe(tabla_sim2.head(50)[columnas_tabla], 
                       use_container_width=True, hide_index=True,
                       column_config={
                           "playerName":"Jugador",
                           "position":"Posición",
                           "rol_desc":"Rol",
                           "nivel_id_{}".format(position_padre):"Nivel",
                           "img_logo": st.column_config.ImageColumn(""
                       ),
                           "logo": st.column_config.ImageColumn(
                               ""
                           ),
                           "adecuacion_total":st.column_config.ProgressColumn(
                                   "{}".format("Similitud"),min_value=20,max_value=99,width="small",
                                   help="SIMILITUD:Mide el grado de distancia entre dos perfiles de jugador.",
                                   format="%f"),
                                       "Performance_{}".format(position_padre):st.column_config.ProgressColumn(
                                               "PER",min_value=20,max_value=99,width="small",
                                               help="PERFORMANCE: Mide el nivel neto del jugador",
                                               format="%f")
                           },height=300)
    
    
    
    st.divider()
    st.markdown("""### :black_circle: Detalle por Medida del Juego""")
    #options_viz=["***Dispersión de Jugadores***","***Mapa de Calor de Prototipos***"]
    #vizo = st.radio(
    #            "Selecciona Opción de Visualización",options_viz,horizontal =True)
    
    c1a,c2a,c3a = st.columns([.15,.45,.4])    
    selectcol = c2a.multiselect(
            "Selecciona medidas a comparar en el gráfico de dispersión",
            list(df_cols[df_cols.position==position_padre].sort_values(by="ponderacion_pct",ascending=False).fancy_name_esp.unique()),
            list(df_cols[df_cols.position==position_padre].sort_values(by="ponderacion_pct",ascending=False).fancy_name_esp.unique()[:2]))
    if len(selectcol)!=2:
        st.write("**:red[Selecciona dos Medidas]**")
    else:
        select_num = c3a.slider(
                    "Selecciona Número de Jugadores a Mostrar",10,df.shape[0],number)
        options=["***Rol***","***Prototipo***"]
        option = c1a.radio(
                    "Selecciona categoría",options,horizontal =True)
        if option==options[0]:
            opt="rol_desc"
        else:
            opt="nivel_id_{}".format(position_padre)
        #col4.markdown("""### :black_circle: Dashboard por Jugador""")
    
        
        df_filtrado = df[
        (df['playerId'].isin(tabla_adecuacion.head(select_num)['playerId'].unique())) |
        (df['teamName'] == teams)
        ]
        
        # Obtener nombres de columna interna (medida) a partir del nombre visible
        x_col = df_cols[df_cols['fancy_name_esp'] == selectcol[0]]['medida'].values[0]
        y_col = df_cols[df_cols['fancy_name_esp'] == selectcol[1]]['medida'].values[0]
        
        # Dibujar gráfico
        fig = pp.scatterplot_plotly(df_filtrado,df_cols, x_col, y_col, opt,teams,position_padre)
        c6,c7 = st.columns([.6,.4])    
        # Mostrar en columna
        c6.plotly_chart(fig, use_container_width=True)
        
        
        c7.plotly_chart(pp.boxplot_xaxisv2_plotly(df_filtrado,
            tss[columnas_tabla].sort_values(by=criterio,ascending=False).index.values[0],
            x_col,
            "rol_desc",
            teams,
            selectcol[0]
            ))
        c7.plotly_chart(pp.boxplot_xaxisv2_plotly(df_filtrado,
            tss[columnas_tabla].sort_values(by=criterio,ascending=False).index.values[0],
            y_col,
            "rol_desc",
            teams,
            selectcol[1],
            False
            ))
    
# Ejecutar
if __name__ == "__main__":
    main()