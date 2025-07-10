# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 13:36:58 2025

@author: aleex
"""

import streamlit as st
import pandas as pd
import json
from PIL import Image
import UTILS_BBDD as ub
import plotly.graph_objects as go

st.set_page_config(page_title="ScoutingLAB - Portada", layout="wide")
@st.cache_resource
def get_conn():
    con = ub.get_conn("config")
    return con

@st.cache_data(ttl=86400)  # 86400 segundos = 24 horas
def get_data():
    conn=get_conn()
    
    df_team = pd.read_sql("select * from fact_ag_team_season", conn)
    df_team=df_team.drop_duplicates()
    df_team= ub.clean_df(df_team)
    dim_team=pd.read_sql("""select * from dim_team""",conn)
    df_cols_team= pd.read_sql("""select * from fact_medida_team""",conn)
    dim_modelo_categoria= pd.read_sql("""select * from dim_modelo_categoria""",conn)
    dim_medida_team = pd.read_sql("""select * from dim_medida_team""",conn)
    df_team=pd.merge(df_team,dim_team[['teamId','season','img_logo']],
                     on=['teamId','season'],how='left')
    dim_competicion=pd.read_sql("""select * from dim_competicion fp
                     """, conn)
    
    return [df_team,df_cols_team,dim_team,dim_medida_team,dim_modelo_categoria,dim_competicion]

def boxplot_xaxisv2_plotly_teams(df, select_pl, col, cluster_col, yaxis_title="", show_legend=True):
    fig = go.Figure()
    colors = ['red', 'purple', 'turquoise', 'orange']
    selected_trace = None

    # Obtener equipo seleccionado
    select_row = df[df.teamName == select_pl]

    # Obtener equipos de la misma competición (en todo el dataframe, no solo dentro del cluster)
    if not select_row.empty:
        comp = select_row.competition.values[0]
        select_comp = df[df.competition == comp]
    else:
        select_comp = pd.DataFrame()

    # Añadir boxplots por cluster
    for i, color in zip(sorted(df[cluster_col].dropna().unique()), colors):
        cluster_df = df[df[cluster_col] == i]

        # Boxplot (sin hover)
        fig.add_trace(go.Box(
            y=cluster_df[col],
            name=str(i),
            marker_color=color,
            line=dict(width=1),
            boxmean='sd',
            boxpoints=False,
            hoverinfo='skip',
            showlegend=show_legend
        ))

    # Añadir puntos de equipos de la misma competición
    if not select_comp.empty:
        for i in sorted(df[cluster_col].dropna().unique()):
            comp_in_cluster = select_comp[select_comp[cluster_col] == i]
            fig.add_trace(go.Scatter(
                x=[str(i)] * len(comp_in_cluster),
                y=comp_in_cluster[col],
                mode='markers',
                marker=dict(size=8, color='black', symbol='circle'),
                text=comp_in_cluster['teamName'],
                hovertemplate="<b>%{text}</b><br>Valor: %{y:.2f}<extra></extra>",
                showlegend=False
            ))

    # Añadir marcador del equipo seleccionado
    if not select_row.empty:
        cluster_val = str(select_row[cluster_col].values[0])
        fig.add_trace(go.Scatter(
            x=[cluster_val],
            y=select_row[col],
            mode='markers',
            marker=dict(size=14, color='lime', symbol='x', line=dict(color='black', width=1)),
            text=select_row['teamName'],
            hovertemplate="<b>%{text}</b><br>Valor: %{y:.2f}<extra></extra>",
            name=select_row['teamName'].values[0],
            showlegend=True
        ))

    # Configuración del eje Y
    yaxis_config = dict(title=yaxis_title)
    if "pct" in col:
        yaxis_config["tickformat"] = ".0%"

    # Layout final
    fig.update_layout(
        title=dict(
            text="{} | Dispersión por Modelo de Juego".format(yaxis_title.upper()),
            x=0.5,
            xanchor='center',
            font=dict(size=13, family='Segoe UI', color='black'),
            y=.99,
        ),
        height=450,
        width=1000,
        template='simple_white',
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
            title=""
        ),
        yaxis=yaxis_config,
        xaxis=dict(
            title="",
            showticklabels=False
        ),
        margin=dict(t=100, r=40, b=40, l=40)
    )

    return fig
# Conexión
conn = get_conn()
df_team,df_cols_team,dim_team,dim_medida_team,dim_modelo_categoria ,dim_competicion= get_data()
# Sidebar
st.sidebar.title("🏁 Portada")
st.sidebar.subheader("Filtros")

comp_opts = list(dim_competicion[dim_competicion.pais_desc.isin(["España","Inglaterra","Italia","Francia","Alemania"])].sort_values(by=["tier_num","pais_id"]).competition.unique())
season_opts = sorted(["2024-2025"])

# Inicializamos session_state con valores si no existen
if "comps" not in st.session_state:
    st.session_state["comps"] = comp_opts[0]

if "seasons" not in st.session_state:
    st.session_state["seasons"] = season_opts[-1]

# Esta función recalcula la lista de equipos validos
def get_team_options():
    return sorted(dim_team[
        (dim_team.competition == st.session_state["comps"]) & 
        (dim_team.season == st.session_state["seasons"])
    ].teamName.unique())

# Inicializamos equipo si no existe o no está en opciones actuales
team_opts = get_team_options()
if "teams" not in st.session_state or st.session_state["teams"] not in team_opts:
    st.session_state["teams"] = team_opts[0] if team_opts else None

# Función callback para competición
def on_change_competition():
    st.session_state["comps"] = st.session_state["select_comps"]
    # Actualizamos lista de equipos
    new_team_opts = get_team_options()
    # Si el equipo actual no está en el nuevo filtro, lo actualizamos
    if st.session_state["teams"] not in new_team_opts:
        st.session_state["teams"] = new_team_opts[0] if new_team_opts else None

# Función callback para temporada
def on_change_season():
    st.session_state["seasons"] = st.session_state["select_seasons"]
    new_team_opts = get_team_options()
    if st.session_state["teams"] not in new_team_opts:
        st.session_state["teams"] = new_team_opts[0] if new_team_opts else None

def on_change_team():
    st.session_state["teams"] = st.session_state["select_teams"]
    return st.session_state["teams"]
# Sidebar: competición
st.sidebar.selectbox(
    "Selecciona Competición",
    options=comp_opts,
    index=comp_opts.index(st.session_state["comps"]),
    key="select_comps",
    on_change=on_change_competition
)

# Sidebar: temporada
st.sidebar.selectbox(
    "Selecciona Temporada",
    options=season_opts,
    index=season_opts.index(st.session_state["seasons"]),
    key="select_seasons",
    on_change=on_change_season
)

# Recalcular equipos filtrados con la competición y temporada actuales
team_opts = sorted(dim_team[
    (dim_team["competition"] == st.session_state.comps) &
    (dim_team["season"] == st.session_state.seasons)
]["teamName"].unique())

selected_team_index = team_opts.index(st.session_state["teams"]) if st.session_state["teams"] in team_opts else 0


st.sidebar.selectbox("Selecciona Equipo", team_opts, 
                     index=selected_team_index,
                     key="select_teams",on_change=on_change_team)

# Ahora tienes las variables con las opciones sincronizadas:
comps = st.session_state["comps"]
seasons = st.session_state["seasons"]
teams = st.session_state["teams"]

# Ejemplo: mostrar logo o info
logo = dim_team[(dim_team.teamName == teams) & (dim_team.season == seasons)].img_logo.values[0]
image_mini=Image.open("Documentación/logo.png")
st.markdown("""
<style>
    /* Reducir margen superior del contenedor principal */
    .block-container {
        padding-top: 3rem;
    }

    /* Ajustar margen del encabezado h1/h2/h3 según corresponda */
    h1, h2, h3 {
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }

    /* Reducir espacio general */
    .st-emotion-cache-1avcm0n {
        padding-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.image(image_mini, width=140)

if seasons:
    df_team = df_team[df_team['season']==seasons]
    if teams:
        
        teamid = dim_team[(dim_team.teamName==teams) & (dim_team.season==seasons)].teamId.values[0]
        modelo_juego=df_team[df_team.teamId==teamid].modelo_id.values[0]
        logo = dim_team[(dim_team.teamName==teams) & (dim_team.season==seasons)].img_logo.values[0]


# Cuerpo
st.title("Bienvenid@ a ScoutingLAB")
col1,col2=st.columns([1.3,.7])
col1.markdown("""
El proyecto ***SCOUTINGLAB*** propone un enfoque cuantitativo del análisis de jugadores, estructurado en torno a dos ejes: **nivel** y **perfil**.
Su metodología se basa en cinco principios:
- **Fundamentar la selección en el modelo de juego del equipo**. Así, se asegura que cualquier incorporación atienda una necesidad real derivada del juego.
- Basar el proceso de identificación en dos métricas de valor cuantificables: el nivel del jugador y su perfil. Ambas deben surgir de medidas relacionadas con el juego, cuya selección se lleva a cabo empleando modelos que establecen su relevancia.
- Desarrollar modelos que permitan segmentar a las entidades principales –jugadores y equipos- en base a las citadas métricas de valor.
- Obtener una entidad analítica que permita, a partir del perfil del jugador, conocer y comparar su capacidad de **ADECUACIÓN** a un modelo de juego.
- Obtener una entidad analítica que mida el **RENDIMIENTO** neto del jugador y valore las **DISTANCIAS** de nivel entre ellos.

[**Saber más**](https://drive.google.com/file/d/1fOr4nmB8YD9AUqSAZw0sQL9svS_011dg/view?usp=sharing)
""")
meto=Image.open("Documentación/metodologia.png")
col2.image(meto)
st.divider()
df_own = df_team[(df_team.teamName==teams) & (df_team.season==seasons)]
col_logo, col_title = st.columns([0.05, 0.95])
with col_logo:
    st.image(logo, width=60)  # Ajusta el tamaño del logo
col_title.markdown("### {} | Modelo de Juego for Fases".format(teams))
kpi1,kpi2,kpi3,kpi4=st.columns([.25,.25,.25,.25])
with kpi1:
    st.markdown("""
    <div style="padding: 10px; border: 1px solid #ccc; border-radius: 10px; background-color: #f9f9f9; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
        <h4 style="margin-bottom:5px;">Estructura</h4>
        <h6>{}</h6>
    </div>
    """.format(dim_modelo_categoria[
        (dim_modelo_categoria.cluster == df_own.cluster_ESTRUCTURA.values[0]) & 
        (dim_modelo_categoria.categoria_id == 5)
    ].modelo_desc.values[0]), unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div style="padding: 10px; border: 1px solid #ccc; border-radius: 10px; background-color: #f9f9f9; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
        <h4 style="margin-bottom:5px;">Defensa</h4>
        <h6>{}</h6>
    </div>
    """.format(dim_modelo_categoria[
        (dim_modelo_categoria.cluster == df_own.cluster_DEFENSA.values[0]) & 
        (dim_modelo_categoria.categoria_id == 1)
    ].modelo_desc.values[0]), unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div style="padding: 10px; border: 1px solid #ccc; border-radius: 10px; background-color: #f9f9f9; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
        <h4 style="margin-bottom:5px;">Construcción</h4>
        <h6>{}</h6>
    </div>
    """.format(dim_modelo_categoria[
        (dim_modelo_categoria.cluster == df_own.cluster_CONSTRUCCION.values[0]) & 
        (dim_modelo_categoria.categoria_id == 2)
    ].modelo_desc.values[0]), unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div style="padding: 10px; border: 1px solid #ccc; border-radius: 10px; background-color: #f9f9f9; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
        <h4 style="margin-bottom:5px;">Ataque</h4>
        <h6>{}</h6>
    </div>
    """.format(dim_modelo_categoria[
        (dim_modelo_categoria.cluster == df_own.cluster_ATAQUE.values[0]) & 
        (dim_modelo_categoria.categoria_id == 3)
    ].modelo_desc.values[0]), unsafe_allow_html=True)
help_uni="UNICIDAD: nº y % de equipos de la misma liga que comparten modelo de juego -general o por fase-"
kpi1.write("")
kpi1.metric("**Unicidad - ESTRUCTURA**".format(comps.upper()), "{:.0%}".format(df_team[(df_team.cluster_ESTRUCTURA==df_own.cluster_ESTRUCTURA.values[0]) & (df_team.competition==comps)].shape[0] / df_team[(df_team.competition==comps)].shape[0]),
          "= que {:.0f} Equipo(s) en {}".format(df_team[(df_team.cluster_ESTRUCTURA==df_own.cluster_ESTRUCTURA.values[0]) & (df_team.competition==comps)].shape[0]-1,comps.upper()),
          border=True, delta_color="off",
          help=help_uni)
kpi2.write("")
kpi2.metric("**Unicidad - DEFENSA**".format(comps.upper()), "{:.0%}".format(df_team[(df_team.cluster_DEFENSA==df_own.cluster_DEFENSA.values[0]) & (df_team.competition==comps)].shape[0] / df_team[(df_team.competition==comps)].shape[0]),
          "= que {:.0f} Equipo(s) en {}".format(df_team[(df_team.cluster_DEFENSA==df_own.cluster_DEFENSA.values[0]) & (df_team.competition==comps)].shape[0]-1,comps.upper()),
          border=True, delta_color="off",help=help_uni)
kpi3.write("")
kpi3.metric("**Unicidad - CONSTRUCCIÓN**".format(comps.upper()), "{:.0%}".format(df_team[(df_team.cluster_CONSTRUCCION==df_own.cluster_CONSTRUCCION.values[0]) & (df_team.competition==comps)].shape[0] / df_team[(df_team.competition==comps)].shape[0]),
          "= que {:.0f} Equipo(s) en {}".format(df_team[(df_team.cluster_CONSTRUCCION==df_own.cluster_CONSTRUCCION.values[0]) & (df_team.competition==comps)].shape[0]-1,comps.upper()),
          border=True, delta_color="off",help=help_uni)
kpi4.write("")
kpi4.metric("**Unicidad - ATAQUE**".format(comps.upper()), "{:.0%}".format(df_team[(df_team.cluster_ATAQUE==df_own.cluster_ATAQUE.values[0]) & (df_team.competition==comps)].shape[0] / df_team[(df_team.competition==comps)].shape[0]),
          "= que {:.0f} Equipo(s) en {}".format(df_team[(df_team.cluster_ATAQUE==df_own.cluster_ATAQUE.values[0]) & (df_team.competition==comps)].shape[0]-1,comps.upper()),
          border=True, delta_color="off",help=help_uni)

kpi1.write("")
kpi1,kpi2,kpi3=st.columns([.3,.3,.3])
kpi2.metric("**UNICIDAD DEL MODELO**".format(comps.upper()), "{:.0%}".format(df_team[(df_team.modelo_id==df_own.modelo_id.values[0]) & (df_team.competition==comps)].shape[0] / df_team[(df_team.competition==comps)].shape[0]),
          "= que {:.0f} Equipo(s) en {}".format(df_team[(df_team.modelo_id==df_own.modelo_id.values[0]) & (df_team.competition==comps)].shape[0]-1,comps.upper()),
          border=True, delta_color="off",help=help_uni)
sisi=[.37,.37,.26]

st.divider()
st.markdown("""### :black_circle: Estructura""",help="Mide la elección de sistemas y ocupación por líneas del campo. Describe la organización base de un equipo.")
st.markdown("#### {}, {} | {}".format(teams.upper(),seasons,dim_modelo_categoria[(dim_modelo_categoria.cluster==df_own.cluster_ESTRUCTURA.values[0]) & (dim_modelo_categoria.categoria_id==5)].modelo_desc.values[0]
                                ))
st.markdown("**{}**".format(dim_modelo_categoria[(dim_modelo_categoria.cluster==df_own.cluster_ESTRUCTURA.values[0]) & (dim_modelo_categoria.categoria_id==5)].modelo_desc_long.values[0])      
            )                                      
ct, ct2, ct3 = st.columns(sisi)

ct, ct3 = st.columns([.74,.26])

selectcol = ct.multiselect(
        "Selecciona medidas de ESTRUCTURA para los gráficos de dispersión",
        list(df_cols_team[df_cols_team.categoria_id==5].sort_values(by="importance_perfil",ascending=False).fancy_name_esp.unique()),
        list(df_cols_team[df_cols_team.categoria_id==5].sort_values(by="importance_perfil",ascending=False).fancy_name_esp.unique()[:2]))
if len(selectcol)!=2:
    st.write("**:red[Selecciona dos Medidas]**")
else:
    ct3.write("")  # Línea en blanco
    ct3.write("")  # Otra más si hace falta
    ct3.write("")  # Otra más si hace falta
    ct3.markdown("**Equipos con mismo modelo en ESTRUCTURA**")
    
    raf, raf2, sim = st.columns(sisi)
    #est_com=ct3.selectbox("Estructura: Competición", options=comp_opts, index=comp_opts.index(st.session_state["comps"]))
    sim.dataframe(df_team[(df_team.cluster_ESTRUCTURA==df_own.cluster_ESTRUCTURA.values[0])][["img_logo","teamName","competition"]],
                  column_config={
                      "teamName":"Equipo",
                      "competition":"Competición",
                      "img_logo": st.column_config.ImageColumn(""
                  )},
                  height=400,use_container_width=True, hide_index=True)
    df_team=pd.merge(df_team,dim_modelo_categoria[dim_modelo_categoria.categoria_id==5][['cluster','modelo_desc']],left_on="cluster_ESTRUCTURA",right_on="cluster")
    raf.plotly_chart(boxplot_xaxisv2_plotly_teams(df_team,df_own.teamName.values[0],
                                                  dim_medida_team[dim_medida_team.fancy_name_esp==selectcol[0]].medida.values[0],
                                                  "modelo_desc",
                                                  selectcol[0]
                                                  ))
    raf2.plotly_chart(boxplot_xaxisv2_plotly_teams(df_team,df_own.teamName.values[0],
                                                  dim_medida_team[dim_medida_team.fancy_name_esp==selectcol[1]].medida.values[0],
                                                  "modelo_desc",
                                                  selectcol[1]
                                                  ))
    df_team.drop(['cluster','modelo_desc'],axis=1,inplace=True)
    
st.divider()
st.markdown("""### :black_circle: Defensa""",help="Define el enfoque de un equipo cuando no tiene el balón y debe recuperarlo o proteger su portería, tanto a nivel posicional como en transición defensiva")
st.markdown("#### {}, {} | {}".format(teams.upper(),seasons,dim_modelo_categoria[(dim_modelo_categoria.cluster==df_own.cluster_DEFENSA.values[0]) & (dim_modelo_categoria.categoria_id==1)].modelo_desc.values[0]
                                ))
st.markdown("**{}**".format(dim_modelo_categoria[(dim_modelo_categoria.cluster==df_own.cluster_DEFENSA.values[0]) & (dim_modelo_categoria.categoria_id==1)].modelo_desc_long.values[0])      
            ) 
ct, ct3 = st.columns([.74,.26])

selectcol = ct.multiselect(
        "Selecciona medidas de DEFENSA para los gráficos de dispersión",
        list(df_cols_team[df_cols_team.categoria_id==1].sort_values(by="importance_perfil",ascending=False).fancy_name_esp.unique()),
        list(df_cols_team[df_cols_team.categoria_id==1].sort_values(by="importance_perfil",ascending=False).fancy_name_esp.unique()[:2]))
if len(selectcol)!=2:
    st.write("**:red[Selecciona dos Medidas]**")
else:
    ct3.write("")  # Línea en blanco
    ct3.write("")  # Otra más si hace falta
    ct3.write("")  # Otra más si hace falta
    ct3.markdown("**Equipos con mismo modelo en DEFENSA**")
    
    raf, raf2, sim = st.columns(sisi)
    #est_com=ct3.selectbox("DEFENSA: Competición", options=comp_opts, index=comp_opts.index(st.session_state["comps"]))
    sim.dataframe(df_team[(df_team.cluster_DEFENSA==df_own.cluster_DEFENSA.values[0])][["img_logo","teamName","competition"]],
                  column_config={
                      "teamName":"Equipo",
                      "competition":"Competición",
                      "img_logo": st.column_config.ImageColumn(""
                  )},
                  height=400,use_container_width=True, hide_index=True)
    df_team=pd.merge(df_team,dim_modelo_categoria[dim_modelo_categoria.categoria_id==1][['cluster','modelo_desc']],left_on="cluster_DEFENSA",right_on="cluster")
    raf.plotly_chart(boxplot_xaxisv2_plotly_teams(df_team,df_own.teamName.values[0],
                                                  dim_medida_team[dim_medida_team.fancy_name_esp==selectcol[0]].medida.values[0],
                                                  "modelo_desc",
                                                  selectcol[0]
                                                  ))
    raf2.plotly_chart(boxplot_xaxisv2_plotly_teams(df_team,df_own.teamName.values[0],
                                                  dim_medida_team[dim_medida_team.fancy_name_esp==selectcol[1]].medida.values[0],
                                                  "modelo_desc",
                                                  selectcol[1]
                                                  ))
    df_team.drop(['cluster','modelo_desc'],axis=1,inplace=True)
    
    
st.divider()
st.markdown("""### :black_circle: Construcción""",help="Define el enfoque de un equipo en fase de creación con balón, tanto a nivel posicional como en transición ofensiva.")
st.markdown("#### {}, {} | {}".format(teams.upper(),seasons,dim_modelo_categoria[(dim_modelo_categoria.cluster==df_own.cluster_CONSTRUCCION.values[0]) & (dim_modelo_categoria.categoria_id==2)].modelo_desc.values[0]
                                ))
st.markdown("**{}**".format(dim_modelo_categoria[(dim_modelo_categoria.cluster==df_own.cluster_CONSTRUCCION.values[0]) & (dim_modelo_categoria.categoria_id==2)].modelo_desc_long.values[0])      
            ) 
ct, ct3 = st.columns([.74,.26])
selectcol = ct.multiselect(
        "Selecciona medidas de CONSTRUCCIÓN para los gráficos de dispersión",
        list(df_cols_team[df_cols_team.categoria_id==2].sort_values(by="importance_perfil",ascending=False).fancy_name_esp.unique()),
        list(df_cols_team[df_cols_team.categoria_id==2].sort_values(by="importance_perfil",ascending=False).fancy_name_esp.unique()[:2]))
if len(selectcol)!=2:
    st.write("**:red[Selecciona dos Medidas]**")
else:
    
    ct3.write("")  # Línea en blanco
    ct3.write("")  # Otra más si hace falta
    ct3.write("")  # Otra más si hace falta
    ct3.markdown("**Equipos con mismo modelo en CONSTRUCCION**")
    
    raf, raf2, sim = st.columns(sisi)
    #est_com=ct3.selectbox("CONSTRUCCION: Competición", options=comp_opts, index=comp_opts.index(st.session_state["comps"]))
    sim.dataframe(df_team[(df_team.cluster_CONSTRUCCION==df_own.cluster_CONSTRUCCION.values[0])][["img_logo","teamName","competition"]],
                  column_config={
                      "teamName":"Equipo",
                      "competition":"Competición",
                      "img_logo": st.column_config.ImageColumn(""
                  )},
                  height=400,use_container_width=True, hide_index=True)
    df_team=pd.merge(df_team,dim_modelo_categoria[dim_modelo_categoria.categoria_id==2][['cluster','modelo_desc']],left_on="cluster_CONSTRUCCION",right_on="cluster")
    raf.plotly_chart(boxplot_xaxisv2_plotly_teams(df_team,df_own.teamName.values[0],
                                                  dim_medida_team[dim_medida_team.fancy_name_esp==selectcol[0]].medida.values[0],
                                                  "modelo_desc",
                                                  selectcol[0]
                                                  ))
    raf2.plotly_chart(boxplot_xaxisv2_plotly_teams(df_team,df_own.teamName.values[0],
                                                  dim_medida_team[dim_medida_team.fancy_name_esp==selectcol[1]].medida.values[0],
                                                  "modelo_desc",
                                                  selectcol[1]
                                                  ))
    df_team.drop(['cluster','modelo_desc'],axis=1,inplace=True)
    
    
st.divider()
st.markdown("""### :black_circle: Ataque""",help="Explica el modo en que el equipo pretende generar peligro sobre la portería rival.")
st.markdown("#### {}, {} | {}".format(teams.upper(),seasons,dim_modelo_categoria[(dim_modelo_categoria.cluster==df_own.cluster_ATAQUE.values[0]) & (dim_modelo_categoria.categoria_id==3)].modelo_desc.values[0]
                                ))
st.markdown("**{}**".format(dim_modelo_categoria[(dim_modelo_categoria.cluster==df_own.cluster_ATAQUE.values[0]) & (dim_modelo_categoria.categoria_id==3)].modelo_desc_long.values[0])      
            ) 
ct, ct3 = st.columns([.74,.26])
selectcol = ct.multiselect(
        "Selecciona medidas de ATAQUE para los gráficos de dispersión",
        list(df_cols_team[df_cols_team.categoria_id==3].sort_values(by="importance_perfil",ascending=False).fancy_name_esp.unique()),
        list(df_cols_team[df_cols_team.categoria_id==3].sort_values(by="importance_perfil",ascending=False).fancy_name_esp.unique()[:2]))
if len(selectcol)!=2:
    st.write("**:red[Selecciona dos Medidas]**")
else:

    ct3.write("")  # Línea en blanco
    ct3.write("")  # Otra más si hace falta
    ct3.write("")  # Otra más si hace falta
    ct3.markdown("**Equipos con mismo modelo en ATAQUE**")
    
    raf, raf2, sim = st.columns(sisi)
    #est_com=ct3.selectbox("ATAQUE: Competición", options=comp_opts, index=comp_opts.index(st.session_state["comps"]))
    sim.dataframe(df_team[(df_team.cluster_ATAQUE==df_own.cluster_ATAQUE.values[0])][["img_logo","teamName","competition"]],
                  column_config={
                      "teamName":"Equipo",
                      "competition":"Competición",
                      "img_logo": st.column_config.ImageColumn(""
                  )},
                  height=400,use_container_width=True, hide_index=True)
    df_team=pd.merge(df_team,dim_modelo_categoria[dim_modelo_categoria.categoria_id==3][['cluster','modelo_desc']],left_on="cluster_ATAQUE",right_on="cluster")
    raf.plotly_chart(boxplot_xaxisv2_plotly_teams(df_team,df_own.teamName.values[0],
                                                  dim_medida_team[dim_medida_team.fancy_name_esp==selectcol[0]].medida.values[0],
                                                  "modelo_desc",
                                                  selectcol[0]
                                                  ))
    raf2.plotly_chart(boxplot_xaxisv2_plotly_teams(df_team,df_own.teamName.values[0],
                                                  dim_medida_team[dim_medida_team.fancy_name_esp==selectcol[1]].medida.values[0],
                                                  "modelo_desc",
                                                  selectcol[1]
                                                  ))