import pandas as pd
import json
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image
from mplsoccer import VerticalPitch,add_image
import matplotlib.patches as patches


sta="config/clas/star1.png"
sta2="config/clas/star2.png"
sta3="config/clas/star3.png"
c="config/clas/circulo.png"
c6="config/clas/numero-6.png"
star = Image.open(sta)
star2 = Image.open(sta2)
star3 = Image.open(sta3)
ci = Image.open(c)
ci6 = Image.open(c6)

def subplotting(ax1,ax2):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')
    ax[0]=ax1
    ax[1]=ax2
    fig.tight_layout()
    plt.show();
        

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_df(df, ranges):
    """scales df[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(df[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = df[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdf = [d]
    for d, (y1, y2) in zip(df[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdf.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdf

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=5):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.8,0.8],polar=True, alpha=0.6,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables,
                                         size=18,
                                         weight='bold',
                                         c='black')

        # [txt.set_rotation(angle-90) for txt, angle 
        #      in zip(text, angles)]
        for ax in axes[1:]:
            ax.tick_params(axis='both', which='major', pad=12,color='grey')
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
            ax.yaxis.grid(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            gridlabel[-1]=""
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
            
        
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='darkmagenta')
    def mplot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='red',alpha=0.55)
    def maxplot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='grey',alpha=0.25)
    def fill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='darkmagenta')
    def mfill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='red')
    def maxfill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='grey')

class ComplexRadar_comp():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=5):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.8,0.8],polar=True, alpha=0.6,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables,
                                         size=10,
                                         weight='bold',
                                         c='black')

        # [txt.set_rotation(angle-90) for txt, angle 
        #      in zip(text, angles)]
        for ax in axes[1:]:
            ax.tick_params(axis='both', which='major', pad=10,color='grey')
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
            ax.yaxis.grid(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            gridlabel[-1]=""
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
            
        
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='darkmagenta')
    def mplot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='red',alpha=0.55)
    def maxplot(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.plot(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='grey',alpha=0.25)
    def fill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='darkmagenta')
    def mfill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='red')
    def maxfill(self, df, *args, **kw):
        sdf = _scale_df(df, self.ranges)
        self.ax.fill(self.angle, np.r_[sdf, sdf[0]], *args, **kw,color='grey')
        
def scatter_xaxisv1(df,select_pl,col,cluster_col,criterio):
    fig,ax=plt.subplots(figsize=(12,6))
    for i,color,r in zip(list(sorted(df[cluster_col].unique())),['red','purple','turquoise','orange'],[.98,.96,.94,.92]):
        print(i)
        p = df[df[cluster_col]==i]
        p2 = p[p.index==select_pl]
        if p2.shape[0]>0:
            p = p[p.index!=p2.index.values[0]]
        plt.scatter(p[col],[r] * p.shape[0], edgecolors="black",
                    c=color,alpha=.7,label='{}'.format(i),
                    s =500)
        ax.axvline(p[col].mean(),0,color=color,alpha=.9,linestyle='--',linewidth=1.5)
        if p2.shape[0]>0:
            plt.scatter(p2[col],[r] * p2.shape[0], 
                    s=500,c="lime",alpha=1,edgecolors="black",marker='X',label='{}'.format(p2.playerName.values[0]))
        #plt.scatter(x[0], y[0], s=sizes[0],marker='X')
        #ax.gca().invert_xaxis()
    # Customize the plot (optional)
    sns.despine()
    #plt.title("No. Clusters para {}: {:.0f}".format(pose,len(df.Cluster.unique())))
    ax.legend(frameon=True,fontsize='12',loc='best')
    #ax.set_xlim([df[criterio].min(),99])
    ax.set_ylim([.875,1])

    plt.gca().set_yticklabels([])
    plt.title("{} por Rol\n".format(criterio),size=16,weight="bold")
    
    #ax.axvline(plf_m.Similarity.mean(),0,color='forestgreen',alpha=.9,linestyle='--',linewidth=2)
    return fig





def scatter_xaxisv1_plotly(df, select_pl, col, cluster_col,position_padre):
    

    fig = go.Figure()
    df[col]=df[col].fillna(0)
    colors = ['red', 'purple', 'turquoise', 'orange']
    y_offsets = [.98, .96, .94, .92]

    # Guarda trazas del jugador seleccionado para añadirlas al final
    selected_traces = []

    for i, color, y_val in zip(df[cluster_col].unique(), colors, y_offsets):
        cluster_df = df[df[cluster_col] == i]
        select_row = cluster_df[cluster_df.index == select_pl]
        rest_df = cluster_df.drop(select_row.index)

        # Puntos normales del cluster
        fig.add_trace(go.Scatter(
            x=rest_df[col],
            y=[y_val] * len(rest_df),
            mode='markers',
            marker=dict(size=16, color=color, line=dict(color='black', width=1)),
            hovertext=rest_df['playerName'],
            hoverinfo='text',
            name=f'{i}'
        ))

        # Línea de media
        fig.add_shape(
            type="line",
            x0=rest_df[col].mean(), x1=rest_df[col].mean(),
            y0=y_val - 0.015, y1=y_val + 0.015,
            line=dict(color=color, width=2, dash='dash'),
        )

        # Guarda la traza del jugador si está en este cluster
        if not select_row.empty:
            selected_traces.append(go.Scatter(
                x=select_row[col],
                y=[y_val],
                mode='markers',
                marker=dict(size=25, color='lime', symbol='x', line=dict(color='black', width=1)),
                hovertext=select_row['playerName'].values,
                hoverinfo='text',
                name=f'{select_row["playerName"].values[0]}'
            ))

    # Añade el jugador seleccionado al final
    for trace in selected_traces:
        fig.add_trace(trace)

    # Configuración final
    fig.update_layout(
        height=280,
        width=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1,
            title="Roles {}".format(position_padre)
        ),
        yaxis=dict(visible=False),
        xaxis=dict(title=""),
        template='simple_white',
        margin=dict(t=40, r=40, b=40, l=40)
    )

    return fig

def boxplot_xaxisv1_plotly(df, select_pl, col, cluster_col, team,posiciones,criterios, position_padre):
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = ['red', 'purple', 'turquoise', 'orange']
    selected_trace = None

    for i, color in zip(sorted(df[cluster_col].unique()), colors):
        cluster_df = df[df[cluster_col] == i]
        select_row = cluster_df[cluster_df.index == select_pl]
        team_rows = cluster_df[cluster_df["teamName"] == team]

        # Boxplot del cluster
        fig.add_trace(go.Box(
            y=cluster_df[col],
            name=f'{i}',
            marker_color=color,
            line=dict(width=1),
            boxmean='sd',
            boxpoints=False,
            hoverinfo='skip'
        ))

        # Puntos del equipo propio (sin añadir a la leyenda)
        if not team_rows.empty:
            fig.add_trace(go.Scatter(
                x=[f'{i}'] * len(team_rows),
                y=team_rows[col],
                mode='markers',
                marker=dict(size=10, color=color, symbol='circle', line=dict(color='black', width=1)),
                hovertext=team_rows['playerName'],
                hoverinfo='text',
                showlegend=False
            ))
        
        if not select_row.empty:
                selected_trace = go.Scatter(
                    x=[f'{i}'],
                    y=select_row[col],
                    mode='markers+text',
                    marker=dict(size=14, color='lime', symbol='x', line=dict(color='black', width=1)),
                    text=select_row['playerName'].values,
                    textposition="top center",
                    name=f'{select_row["playerName"].values[0]}',
                    showlegend=False
                )

    if selected_trace:
        fig.add_trace(selected_trace)

    fig.update_layout(
                height=420,
                width=1000,
                template='simple_white',
                title=dict(text="{} - {} por {}".format(posiciones,criterios, cluster_col.split("_")[0].title()),
                                        x=0.5,
                        y=1,  # <-- más arriba
                        xanchor='center',
                        yanchor='top',
                        font=dict(size=15)
                    ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="right",
                    x=1,
                    title="Roles {}".format(position_padre)
                ),
                yaxis=dict(title=""),
                xaxis=dict(
                    title="", 
                    showticklabels=False
                ),
                margin=dict(t=80, r=40, b=40, l=40)  # subo el top margin un poco por el título
            )

    return fig

def boxplot_xaxisv2_plotly(df, select_pl, col, cluster_col, team, yaxis_title="", show_legend=True):


    fig = go.Figure()
    
    colors = ['red', 'purple', 'turquoise', 'orange']
    selected_trace = None

    for i, color in zip(sorted(df[cluster_col].dropna().unique()), colors):
        cluster_df = df[df[cluster_col] == i]
        select_row = cluster_df[cluster_df.index == select_pl]
        team_rows = cluster_df[cluster_df["teamName"] == team]

        # Boxplot del cluster
        fig.add_trace(go.Box(
            y=cluster_df[col],
            name=str(i),
            marker_color=color,
            line=dict(width=1),
            boxmean='sd',
            boxpoints=False,
            showlegend=show_legend,
            hoverinfo='skip'
        ))

        # Puntos del equipo propio (sin leyenda)
        if not team_rows.empty:
            fig.add_trace(go.Scatter(
                x=[str(i)] * len(team_rows),
                y=team_rows[col],
                mode='markers',
                marker=dict(size=10, color=color, symbol='circle', line=dict(color='black', width=1)),
                hovertext=team_rows['playerName'],
                hoverinfo='text',
                showlegend=False
            ))

        # Jugador seleccionado
        if not select_row.empty:
            selected_trace = go.Scatter(
                x=[str(i)],
                y=select_row[col],
                mode='markers+text',
                marker=dict(size=14, color='lime', symbol='x', line=dict(color='black', width=1)),
                text=select_row['playerName'].values,
                textposition="top center",
                name=str(select_row['playerName'].values[0]),
                showlegend=False
            )

    if selected_trace:
        fig.add_trace(selected_trace)

    # Layout
    fig.update_layout(
        height=350,
        width=1000,
        template='simple_white',
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            title=f"{cluster_col.capitalize()}" if show_legend else ""
        ),
        yaxis=dict(title=yaxis_title),
        xaxis=dict(title=""),
        margin=dict(t=40, r=40, b=40, l=40)
    )

    return fig

def sradar(player,data,comp,mins, maxs,col_radar, position_padre,dim_player,dim_team,s=None,nombres=None):
    err=0
    rc = data[(data['playerName_id']==player) & (data.season==s)]
    
    
    perf = 'Performance_{}'.format(position_padre)
    #comp = rc.iloc[0]['Competicion']
    
    img = dim_player[dim_player.playerId==data[(data['playerName_id']==player) & (data.season==s)].playerId.values[0]].logo.values[0]
    
    img2 = dim_team[dim_team.teamName==data[data.playerName_id==player].teamName.values[0]].img_logo.values[0]
    
    title = ['playerName','teamName','competition',
             "minutes",perf]
    
    try:
        response = requests.get("{}".format(dim_team[dim_team.teamName==data[data.playerName_id==player].teamName.values[0]].img_logo.values[0]))
        img = Image.open(BytesIO(response.content))
        
    except:
        err+=1
        print('Could not load team photo')
        pass
    try: 
        response = requests.get("{}".format(dim_player[dim_player.playerName_id==player].logo.values[0]))
        img2 = Image.open(BytesIO(response.content))
        
    except:
        err+=1
        print('Could not load player photo')
        pass
    try:
        age = rc.age.values[0]
    except:
        age=""
    league = rc.competition.values[0]
    try:
        hei = rc.height.values[0]
        wei = rc.weight.values[0]
    except:
        hei=""
        wei=""
    minutes = rc.minutes.values[0]
    pos=rc.position.values[0]
    if position_padre=="CB":
        plp=None
        ini = None
        metrica=None
    elif position_padre=="GK":
        plp = rc["Performance_PARADAS_{}".format(position_padre)].values[0]
        ini = "PAR"
        metrica = "PARADAS"
    else:
        plp = rc["Performance_ATAQUE_{}".format(position_padre)].values[0]
        ini = "ATA"
        metrica = "ATAQUE"
    plc = rc["Performance_CONSTRUCCION_{}".format(position_padre)].values[0]
    pld = rc["Performance_DEFENSA_{}".format(position_padre)].values[0]
    
    
    if nombres:
        changer = nombres
    else:
        changer = [i.replace("_p90","").replace("_padj","").replace("_pct","%").replace("_"," ").title() for i in col_radar]
    
    rc= rc.set_index(title)[col_radar]
    att = list(rc)
    rc.columns = changer
    
    values = rc.iloc[0].tolist()
    values = tuple(values)
    dmean = comp
    print(col_radar)
    if player in list(dmean.prototipo_id):
        pass
    else:
        dmean=pd.concat([dmean,data[data['playerName_id']==player][title+col_radar]])
    ranges=[]
    for i in att:
        if "_pct" not in i:
            ranges.append((data[i].min(),data[i].max()))
        else:
            ranges.append((0,1))
        
    dmean = dmean[dmean['prototipo_id']==mins]
    dmean = dmean[col_radar]
    dmean.columns = changer
    dmax = comp[comp['prototipo_id']==maxs]
    dmax = dmax[col_radar]
    dmax.columns = changer
    mean = dmean.values[0].tolist()
    mean = tuple(mean)
    dmax = dmax.values[0].tolist()
    dmax= tuple(dmax)
              
    
    fig1 = plt.figure(figsize=(18, 20))

    # RADAR
    radar = ComplexRadar(fig1,changer, ranges)
    radar.plot(values)
    radar.mplot(mean)
    radar.maxplot(dmax)
    #plt.style.use('seaborn-white')
    radar.fill(values, alpha=0.5)
    radar.mfill(mean, alpha=0.2)
    radar.maxfill(dmax, alpha=0.2)
    plt.figtext(0.078,0.98,'{}'.format("Radar/  "),size=20,weight='bold',color='dimgrey',fontname="Segoe UI")
    kk = rc.index[0][0]
    plt.figtext(0.13,0.978,'{} | {:.0f}'.format(kk,rc.index[0][-1]),size=40,weight='bold',color='darkmagenta',
                fontname="Segoe UI")
    if s:
        plt.figtext(0.13,0.957,'{} | Temporada {}'.format(league.title(),s),size=24,weight='bold',color='black',fontname="Segoe UI")
    else:
        plt.figtext(0.13,0.957,'{}'.format(league.title()),size=24,weight='bold',color='black',fontname="Segoe UI")
    plt.figtext(0.13,0.94,'{} | {:.0f} años | {:.0f} Minutos | {:.0f}cm'.format(
                    pos,age,minutes,hei                                
        ),
                size=20,fontname="Segoe UI")
    
    
    #plt.figtext(0.75,0.1025,'.',color="red",weight='bold',ha='center',size=90,alpha=0.8)
    
    plt.figtext(0.15,0.14,'{} - DIS {}'.format(league,mins.replace("_"," ").replace("All","").strip().upper()),
                color="red",ha='center',size=20,weight="bold",fontname="Segoe UI")
    if len(s)==1:
        pass
    else:
        #plt.figtext(0.75,0.0725,'.',color="grey",weight='bold',ha='center',size=90,alpha=0.25)
        plt.figtext(0.85,0.14,'{} - DIS {}'.format(league,maxs.replace("_"," ").replace("All","").strip().upper()),
                color="darkgrey",ha='center',size=20,weight="bold",fontname="Segoe UI")
    plt.figtext(0.5,0.04,'Datos normalizados por 90 minutos\nMétricas no porcentuales ajustadas a la posesión del equipo (salvo xG y xA)',ha='center',size=16)
    norm = mcolors.Normalize(vmin=-.5, vmax=.5)
    cmap = cm.get_cmap('RdYlGn')  # Puedes usar otros como 'viridis', 'coolwarm', etc.
    
    plt.figtext(0.7, 0.97, '{:.0f}'.format(pld),
        fontsize=60,weight='bold',
        ha='center', va='center',
        )
    plt.figtext(0.8, 0.97, '{:.0f}'.format(plc),
        fontsize=60,weight='bold',
        ha='center', va='center',
        )
    if plp:
        plt.figtext(0.9, 0.97, '{:.0f}'.format(plp),
            fontsize=60,weight='bold',
            ha='center', va='center',
            )
    
    plt.figtext(.7,.94,"DEF",fontsize=30,weight='bold',
    ha='center', va='center')
    plt.figtext(.8,.94,"CON",fontsize=30,weight='bold',
    ha='center', va='center')
    if metrica:
        plt.figtext(.9,.94,f"{ini}",fontsize=30,weight='bold',
        ha='center', va='center')
    
    
    plt.figtext(0.15, 0.12, '{:.0%}'.format(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_Performance_Promedio_All_{}".format(position_padre,position_padre)].values[0]),
        fontsize=30,weight='bold',
        ha='center', va='center',
        bbox=dict(facecolor=cmap(norm(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_Performance_Promedio_All_{}".format(position_padre,position_padre)].values[0])),
                  alpha=0.6, edgecolor='none',boxstyle='round,pad=0.2'))
    if metrica:
        plt.figtext(0.2, 0.09, '{:.0%}'.format(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_{}_Promedio_All".format(position_padre,metrica)].values[0]),
        fontsize=20,weight='bold',
        ha='center', va='center',
        bbox=dict(facecolor=cmap(norm(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_{}_Promedio_All".format(position_padre,metrica)].values[0])), alpha=0.6, edgecolor='none',boxstyle='round,pad=0.2'))
    plt.figtext(0.1, 0.09, '{:.0%}'.format(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_DEFENSA_Promedio_All".format(position_padre)].values[0]),
        fontsize=20,weight='bold',
        ha='center', va='center',
        bbox=dict(facecolor=cmap(norm(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_DEFENSA_Promedio_top6".format(position_padre)].values[0])), alpha=0.6, edgecolor='none',boxstyle='round,pad=0.2'))
    plt.figtext(0.15, 0.09, '{:.0%}'.format(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_CONSTRUCCION_Promedio_All".format(position_padre)].values[0]),
        fontsize=20,weight='bold',
        ha='center', va='center',
        bbox=dict(facecolor=cmap(norm(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_CONSTRUCCION_Promedio_All".format(position_padre)].values[0])), alpha=0.6, edgecolor='none',boxstyle='round,pad=0.2'))
    if metrica:
        plt.figtext(.2,.07,f"{ini}",fontsize=20,weight='bold',
                    ha='center', va='center')
    plt.figtext(.1,.07,"DEF",fontsize=20,weight='bold',
    ha='center', va='center')
    plt.figtext(.15,.07,"CON",fontsize=20,weight='bold',
    ha='center', va='center')
    plt.figtext(0.85, 0.12, '{:.0%}'.format(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_Performance_Top_All_{}".format(position_padre,position_padre)].values[0]),
        fontsize=30,weight='bold',
        ha='center', va='center',
        bbox=dict(facecolor=cmap(norm(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_Performance_Top_All_{}".format(position_padre,position_padre)])), alpha=0.6, edgecolor='none',boxstyle='round,pad=0.2'))
    if metrica:
        plt.figtext(0.9, 0.09, '{:.0%}'.format(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_{}_Top_All".format(position_padre,metrica)].values[0]),
        fontsize=20,weight='bold',
        ha='center', va='center',
        bbox=dict(facecolor=cmap(norm(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_{}_Top_top6".format(position_padre,metrica)])), alpha=0.6, edgecolor='none',boxstyle='round,pad=0.2'))
    plt.figtext(0.8, 0.09, '{:.0%}'.format(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_DEFENSA_Top_All".format(position_padre,metrica)].values[0]),
        fontsize=20,weight='bold',
        ha='center', va='center',
        bbox=dict(facecolor=cmap(norm(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_DEFENSA_Top_All".format(position_padre)])), alpha=0.6, edgecolor='none',boxstyle='round,pad=0.2'))
    plt.figtext(0.85, 0.09, '{:.0%}'.format(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_CONSTRUCCION_Top_All".format(position_padre)].values[0]),
        fontsize=20,weight='bold',
        ha='center', va='center',
        bbox=dict(facecolor=cmap(norm(data[(data.playerName_id==player) & (data.season==s)]["DIST_{}_CONSTRUCCION_Top_All".format(position_padre)].values[0])), alpha=0.6, edgecolor='none',boxstyle='round,pad=0.2'))
    if metrica:
        plt.figtext(.9,.07,f"{ini}",fontsize=20,weight='bold',
        ha='center', va='center')
    plt.figtext(.8,.07,"DEF",fontsize=20,weight='bold',
    ha='center', va='center')
    plt.figtext(.85,.07,"CON",fontsize=20,weight='bold',
    ha='center', va='center')
    star = Image.open("config/clas/star1.png")

    if rc.index[0][-1] < comp[comp.prototipo_id=="Nivel Medio_All"].Performance.values[0]:
        
        add_image(ci, fig1, left=0.03, bottom=0.91, width=0.02,height=0.02)
    else:
        #if rc.index[0][-1] < comp[comp.prototipo_id=="Nivel Medio_All"].Performance.values[0]:
            #add_image(ci, fig1, left=0.03, bottom=0.91, width=0.02,height=0.02)
            #add_image(star, fig1, left=0.03, bottom=0.91, width=0.02,height=0.02)
        #else:
            if rc.index[0][-1] < comp[comp.prototipo_id=="Nivel Alto_All"].Performance.values[0]: 
                
                #add_image(star, fig1, left=0.03, bottom=0.91, width=0.02,height=0.02)
                #add_image(star, fig1, left=0.02, bottom=0.91, width=0.02,height=0.02)
                add_image(star, fig1, left=0.04, bottom=0.91, width=0.02,height=0.02)
            else:
                if rc.index[0][-1] < comp[comp.prototipo_id=="Nivel Top_All"].Performance.values[0]:
                    add_image(star, fig1, left=0.02, bottom=0.91, width=0.02,height=0.02)
                    add_image(star, fig1, left=0.04, bottom=0.91, width=0.02,height=0.02)
                else:
                    add_image(star, fig1, left=0.01, bottom=0.91, width=0.02,height=0.02)
                    add_image(star, fig1, left=0.03, bottom=0.91, width=0.02,height=0.02)
                    add_image(star, fig1, left=0.05, bottom=0.91, width=0.02,height=0.02)
    if rc.index[0][-1] > comp[comp.prototipo_id=="Promedio_top6"].Performance.values[0]:
        add_image(ci6, fig1, left=0.02, bottom=0.86, width=0.04,height=0.04)
    try:
        
        add_image(plt.imshow(img2), fig1, left=0, bottom=0.93, width=0.08,height=0.08)
    except:
        pass
    try:
        add_image(img, fig1, left=0, bottom=0.93, width=0.08,height=0.08)
    except:
        pass
    #fig1.savefig(os.path.join("Análisis IN",'Output')+'/'+'Radar_{}_{}'.format(player.replace(' ','_'),s),dpi=90)
    return fig1;

    
def sradar_comp(player1,player2,data,comp,mins, maxs,col_radar, position_padre, dim_player,dim_team,s1=None,s2=None,nombres=None):
    err=0
    rc1 = data[(data['playerName_id']==player1) & (data.season==s1)]
    rc2 = data[(data['playerName_id']==player2) & (data.season==s2)]
    
    
    perf = 'Performance_{}'.format(position_padre)
    #comp = rc.iloc[0]['Competicion']
    
        
    try:
        response = requests.get("{}".format(dim_team[dim_team.teamName==data[data.playerName_id==player1].teamName.values[0]].img_logo.values[0]))
        img = Image.open(BytesIO(response.content))
        
    except:
        err+=1
        print('Could not load team photo')
        pass
    
    try:
        response = requests.get("{}".format(dim_team[dim_team.teamName==data[data.playerName_id==player2].teamName.values[0]].img_logo.values[0]))
        img2 = Image.open(BytesIO(response.content))
        
    except:
        err+=1
        print('Could not load team photo')
        pass
    title = ['playerName','teamName','competition',
             'minutes',perf]
    

    
    league1 = rc1.competition.values[0]
    try:
        age1 = rc1.age.values[0]
    except:
        age1=""
    
    try:
        hei1 = rc1.height.values[0]
        wei1 = rc1.weight.values[0]
    except:
        hei1=""
        wei1=""
    minutes1 = rc1.minutes.values[0]
    pos1=rc1.position.values[0]
    if position_padre=="CB":
        plp1=None
        ini1 = None
        metrica1=None
    elif position_padre=="GK":
        plp1 = rc1["Performance_PARADAS_{}".format(position_padre)].values[0]
        ini1 = "PAR"
        metrica1 = "PARADAS"
    else:
        plp1 = rc1["Performance_ATAQUE_{}".format(position_padre)].values[0]
        ini1 = "ATA"
        metrica1 = "ATAQUE"
    plc1 = rc1["Performance_CONSTRUCCION_{}".format(position_padre)].values[0]
    pld1 = rc1["Performance_DEFENSA_{}".format(position_padre)].values[0]
    
    
    
    if nombres:
        changer = nombres
    else:
        changer = [i.replace("_p90","").replace("_padj","").replace("_pct","%").replace("_"," ").title() for i in col_radar]
    
    rc1 = rc1.set_index(title)[col_radar]
    att = list(rc1)
    ranges=[]
    for i in att:
        if "_pct" not in i:
            ranges.append((data[i].min(),data[i].max()))
        else:
            ranges.append((0,1))
    rc1.columns = changer
    
    values = rc1.iloc[0].tolist()
    values = tuple(values)
          
    
    fig1 = plt.figure(figsize=(12, 14))

    # RADAR
    radar = ComplexRadar_comp(fig1,changer, ranges)
    radar.plot(values)

    #plt.style.use('seaborn-white')
    radar.fill(values, alpha=0.5)

    plt.figtext(0.078,0.98,'{}'.format("Radar/  "),size=12,weight='bold',color='dimgrey',fontname="Segoe UI")
    kk = rc1.index[0][0]
    plt.figtext(0.13,0.98,'{} | {:.0f}'.format(kk,rc1.index[0][-1]),size=20,weight='bold',color='darkmagenta',
                fontname="Segoe UI")
    if s1:
        plt.figtext(0.13,0.957,'{} | Temporada {}'.format(league1.title(),s1),size=13,weight='bold',color='black',fontname="Segoe UI")
    else:
        plt.figtext(0.13,0.957,'{}'.format(league1.title()),size=13,weight='bold',color='black',fontname="Segoe UI")
    plt.figtext(0.13,0.94,'{} | {:.0f} años | {:.0f} Minutos | {:.0f}cm'.format(pos1,
                    age1,minutes1,hei1,wei1                              
        ),
                size=11,fontname="Segoe UI")
    
    
    #plt.figtext(0.75,0.1025,'.',color="red",weight='bold',ha='center',size=90,alpha=0.8)
    


    plt.figtext(0.5,0.1,'Datos normalizados por 90 minutos\nMétricas no porcentuales ajustadas a la posesión del equipo (salvo xG y xA)',ha='center',size=8)
    if position_padre!='CB':
        plt.figtext(0.3, 0.9, '{:.0f}'.format(plp1),
            fontsize=28,weight='bold',
            ha='center', va='center',
            )
    plt.figtext(0.1, 0.9, '{:.0f}'.format(pld1),
        fontsize=28,weight='bold',
        ha='center', va='center',
        )
    plt.figtext(0.2, 0.9, '{:.0f}'.format(plc1),
        fontsize=28,weight='bold',
        ha='center', va='center',
        )
    if position_padre!='CB':
        if position_padre=="GK":
            plt.figtext(.3,.88,"PAR",fontsize=18,weight='bold',
            ha='center', va='center')
        else:
            plt.figtext(.3,.88,"ATA",fontsize=18,weight='bold',
            ha='center', va='center')
    plt.figtext(.1,.88,"DEF",fontsize=18,weight='bold',
    ha='center', va='center')
    plt.figtext(.2,.88,"CON",fontsize=18,weight='bold',
    ha='center', va='center')
    
    
    
    try:
        
        add_image(img, fig1, left=0, bottom=0.93, width=0.08,height=0.08)

    except:
        pass
    
    
    if rc1.index[0][-1] < comp[comp.prototipo_id=="Promedio_All"].Performance.values[0]:
        pass
    else:
        if rc1.index[0][-1] < comp[comp.prototipo_id==mins].Performance.values[0]:
            add_image(ci, fig1, left=0.03, bottom=0.91, width=0.02,height=0.02)
        else:
            if rc1.index[0][-1] < comp[comp.prototipo_id=="Nivel Alto_top6"].Performance.values[0]: 
                
                add_image(star, fig1, left=0.03, bottom=0.91, width=0.02,height=0.02)
            else:
                if rc1.index[0][-1] < comp[comp.prototipo_id=="Nivel Top_top6"].Performance.values[0]:
                    add_image(star, fig1, left=0.02, bottom=0.91, width=0.02,height=0.02)
                    add_image(star, fig1, left=0.04, bottom=0.91, width=0.02,height=0.02)
                else:
                    add_image(star, fig1, left=0.01, bottom=0.91, width=0.02,height=0.02)
                    add_image(star, fig1, left=0.03, bottom=0.91, width=0.02,height=0.02)
                    add_image(star, fig1, left=0.05, bottom=0.91, width=0.02,height=0.02)
                    

    
    rc1=rc2
    s1=s2
    
    league1 = rc1.competition.values[0]
    try:
        age1 = rc1.age.values[0]
    except:
        age1=""
    
    try:
        hei1 = rc1.height.values[0]
        wei1 = rc1.weight.values[0]
    except:
        hei1=""
        wei1=""
    minutes1 = rc1.minutes.values[0]
    pos1=rc1.position.values[0]
    if position_padre=="CB":
        plp1=None
        ini1 = None
        metrica1=None
    elif position_padre=="GK":
        plp1 = rc1["Performance_PARADAS_{}".format(position_padre)].values[0]
        ini1 = "PAR"
        metrica1 = "PARADAS"
    else:
        plp1 = rc1["Performance_ATAQUE_{}".format(position_padre)].values[0]
        ini1 = "ATA"
        metrica1 = "ATAQUE"
    plc1 = rc1["Performance_CONSTRUCCION_{}".format(position_padre)].values[0]
    pld1 = rc1["Performance_DEFENSA_{}".format(position_padre)].values[0]
    
    
    
    
    if nombres:
        changer = nombres
    else:
        changer = [i.replace("_p90","").replace("_padj","").replace("_pct","%").replace("_"," ").title() for i in col_radar]
    
    rc1 = rc1.set_index(title)[col_radar]
    att = list(rc1)
    rc1.columns = changer
    
    values = rc1.iloc[0].tolist()
    values = tuple(values)
    
        
    ranges=[]
    for i in att:
        ranges.append((data[i].min(),data[i].max()))
        
    st=.52
    #fig1 = plt.figure(figsize=(18, 20))

    # RADAR
    #radar = ComplexRadar(fig1,changer, ranges)
    radar.mplot(values)

    #plt.style.use('seaborn-white')
    radar.mfill(values, alpha=0.4)

    plt.figtext(st+0.078,0.98,'{}'.format("Radar/  "),size=12,weight='bold',color='dimgrey',fontname="Segoe UI")
    kk = rc1.index[0][0]
    plt.figtext(st+0.13,0.98,'{} | {:.0f}'.format(kk,rc1.index[0][-1]),size=20,weight='bold',color='red',
                fontname="Segoe UI")
    if s1:
        plt.figtext(st+0.13,0.957,'{} | Temporada {}'.format(league1.title(),s1),size=13,weight='bold',color='black',fontname="Segoe UI")
    else:
        plt.figtext(st+0.13,0.957,'{}'.format(league1.title()),size=13,weight='bold',color='black',fontname="Segoe UI")
    plt.figtext(st+0.13,0.94,'{} | {:.0f} años | {:.0f} Minutos | {:.0f}cm'.format(pos1,
                    age1,minutes1,hei1,wei1                              
        ),
                size=11,fontname="Segoe UI")
    
    
    #plt.figtext(0.75,0.1025,'.',color="red",weight='bold',ha='center',size=90,alpha=0.8)
    
    if position_padre!='CB':
        plt.figtext(0.9, 0.9, '{:.0f}'.format(plp1),
            fontsize=28,weight='bold',
            ha='center', va='center',
            )
    plt.figtext(0.7, 0.9, '{:.0f}'.format(pld1),
        fontsize=28,weight='bold',
        ha='center', va='center',
        )
    plt.figtext(0.8, 0.9, '{:.0f}'.format(plc1),
        fontsize=28,weight='bold',
        ha='center', va='center',
        )
    if position_padre!='CB':
        if position_padre=="GK":
            plt.figtext(.9,.88,"PAR",fontsize=18,weight='bold',
            ha='center', va='center')
        else:
            plt.figtext(.9,.88,"ATA",fontsize=18,weight='bold',
            ha='center', va='center')
    
    plt.figtext(.7,.88,"DEF",fontsize=18,weight='bold',
    ha='center', va='center')
    plt.figtext(.8,.88,"CON",fontsize=18,weight='bold',
    ha='center', va='center')
    
    
    try:
        
        add_image(img2, fig1, left=st, bottom=0.93, width=0.08,height=0.08)
    except:
        pass
    

    if rc1.index[0][-1] < comp[comp.prototipo_id=="Promedio_All"].Performance.values[0]:
        pass
    else:
        if rc1.index[0][-1] < comp[comp.prototipo_id==mins].Performance.values[0]:
            add_image(ci, fig1, left=st+0.03, bottom=0.91, width=0.02,height=0.02)
        else:
            if rc1.index[0][-1] < comp[comp.prototipo_id=="Nivel Alto_top6"].Performance.values[0]: 
                
                add_image(star, fig1, left=st+0.02, bottom=0.91, width=0.02,height=0.02)
            else:
                if rc1.index[0][-1] < comp[comp.prototipo_id=="Nivel Top_top6"].Performance.values[0]:
                    add_image(star, fig1, left=st+0.02, bottom=0.91, width=0.02,height=0.02)
                    add_image(star, fig1, left=st+0.04, bottom=0.91, width=0.02,height=0.02)
                else:
                    add_image(star, fig1, left=st+0.01, bottom=0.91, width=0.02,height=0.02)
                    add_image(star, fig1, left=st+0.03, bottom=0.91, width=0.02,height=0.02)
                    add_image(star, fig1, left=st+0.05, bottom=0.91, width=0.02,height=0.02)
    

    #fig1.savefig(os.path.join(ruta,'Output','Players')+'/'+'Radar_{}'.format(player.replace(' ','_')),dpi=90)
    return fig1

def plot_percentiles(player,data,medidas,s,perf,posiciones,mins=None,maxs=None):
    err=0
    rc = data[(data['playerName_id']==player) & (data['season']==s)]
    idx=rc.reset_index()["index"].values[0]
    
    
    

    comp = rc.iloc[0]['competition']
    rc=rc[rc.competition==comp]
    title = ['playerName_id','teamName','minutes','competition',
             perf]
    
    changer = medidas["fancy_name_esp"].unique()
    col_radar = medidas['medida'].unique()
    med= pd.merge(medidas,pd.DataFrame(col_radar,columns=["medida"]).reset_index(),on="medida",how="left")
    ks =list(med[med.calculo_sn==1].sort_values(by="index").categoria_desc.values)
    rc= rc.set_index(title)[col_radar]
    # PERCENTILES
    k= data[title+list(col_radar)]

    k.columns = title+list(changer)
    col_radar_r=[]
    for i in changer:
        #lp[lp['Pos']==i]
        x=i+'pc'
        y=i+'rk'
        k[x] = k[i].rank(pct = True)*100
        k[i+'pc'] = round(k[i+'pc'],0)
        k = k.sort_values(by=i,ascending=False)
        index = range(1,k.shape[0]+1)
        k[i+'rk'] = index
        col_radar_r.append(x)

    k=k[(k['playerName_id']==player) & (k.index==idx)]
    k = k.set_index('playerName_id').iloc[:,4:]
    k_transposed = k.T
    
    

    x=k_transposed.loc[col_radar_r]
    values = list(x[player])
    colors=[]
    for i in values:
        if i<10:
            colors.append("#ad0000")
        elif 20>i>=10:
            colors.append("#d73027")
        elif 30>i>=20:
            colors.append("#f46d43")
        elif 40>i>=30:
            colors.append("#fdae61")
        elif 50>i>=40:
            colors.append("#fee08b")
        elif 65>i>=50:
            colors.append("#f5f10a")
        elif 80>i>=65:
            colors.append("#a6d96a")
        elif 95>i>=80:
            colors.append("#10cc1d")
        elif 100>=i>=95:
            colors.append("#1a9850")
    x['colors']=colors
    
    fig2,ax = plt.subplots(figsize=(10,12.5))
    ax.tick_params(left = False, right = False, top=False)

    sns.barplot(x=player,y=col_radar_r,data=x,palette=colors,alpha=1,edgecolor='white',linewidth=1.5,
                    )
    
    plt.xticks(np.arange(0,100,20))
    
    plt.xlabel('')
    ax.set_title('{} - Percentiles en Medidas Clave vs. {}\n'.format(posiciones,"Ligas Seleccionadas"),size=22,fontname="Segoe UI",
                 ha='center',weight='bold')
    
    ax.set_yticklabels(changer)
    
    for item,i in zip(plt.gca().yaxis.get_ticklabels(),ks):
        #c = colores[i]
        item.set_fontsize(16)
        #item.set_backgroundcolor(c)
        item.set_fontweight('bold')
    
    
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    #fig2.spines['left'].set_color('none')
    
    #fig2.spines['left'].set_smart_bounds(True)
    #fig2.spines['top'].set_smart_bounds(True)
    rects = ax.patches
    y=[]
    
    for rect,i in zip(rects,col_radar_r):
        j = i[:-2]
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        y.append(y_value)
    
        # Number of points between bar and label. Change to your liking.
        space = 3
        # Vertical alignment for positive values
        ha = 'left'
    
        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label at right
            ha = 'right'
    
        # Use X value as label and format number with one decimal place
        label = "{:.0f}  ".format(x_value)
        if "%pc" in i:
            label2 = "{:.0%} ({:.0f}º)".format(float(k[j]),float(k[j+'rk']))
        else:
            label2 = "{:.2f} ({:.0f}º)".format(float(k[j]),float(k[j+'rk']))
    
        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(space, 0),          # Horizontally shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            va='center',                # Vertically center label
            ha=ha,
            size=16,
            weight='bold')                      # Horizontally align label differently for
                                        # positive and negative values.
                                        
        plt.annotate(
        label2,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space*11, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha,
        size=14,
        )                      # Horizontally align label differently for
                                    # positive and negative values.
    #plt.tight_layout()
    #add_image(img, fig2, left=.2, bottom=0.9, width=0.06,height=0.06)
    #cat_medidas = defaultdict(list)
    #for i, cat in enumerate(ks):
    #    cat_medidas[cat[:3]].append(i)
    
    # 2. Añadir texto por categoría
    #for cat, indices in cat_medidas.items():
    #    y_medio = np.mean([y[i] for i in indices])  # y[] ya tiene la coordenada vertical de cada barra
    #    ax.text(-37, y_medio, cat.upper(), fontsize=20, fontweight='bold', va='center', ha='right', color='black')
        
    return fig2 

def mapa_calor_prototipos(df_jugadores, df_prototipos, prototipos, medidas, accro, referencia, s=False, es_jugador=True):


    # Unir ambos dataframes
    df_jugadores["ps_id"] = df_jugadores["playerName"] + "\n" + df_jugadores["season"]
    df_full = pd.concat([
        df_jugadores[(df_jugadores.playerName == referencia) & (df_jugadores.season == s)],
        df_prototipos[df_prototipos.prototipo_id.isin(prototipos)]
    ], ignore_index=True)
    df_full['prototipo_id'] = df_full['prototipo_id'].str.replace("All", df_prototipos.competition.values[0])
    df_full["ps_id"] = np.where(df_full.prototipo_id.notna(), df_full.prototipo_id, df_full.ps_id)

    df_diff = df_full.set_index('ps_id').copy()

    # Obtener la fila de referencia
    if es_jugador:
        ref_row = df_diff[(df_diff.playerName == referencia) & (df_diff.season == s)]
    else:
        ref_row = df_diff[df_diff.index == referencia]
        df_diff = df_diff[df_diff.index != referencia]

    # Calcular diferencias
    df_diff = df_diff[medidas]
    df_diff = df_diff[df_diff.index != ref_row.index[0]]
    for col in df_diff.columns:
        if "prevented" in col:
            df_diff[col] = -1 * (df_diff[col] - ref_row[col].values[0])
        else:
            df_diff[col] = -1 * (df_diff[col] - ref_row[col].values[0]) / (ref_row[col].values[0])

    df_final = pd.concat([ref_row[medidas], df_diff])
    rename_cols = dict(zip(medidas, accro))
    df_final.rename(rename_cols, axis=1, inplace=True)
    df_final.index = [i.replace("_", "\n") for i in df_final.index]

    # 🔁 Transponer el dataframe
    df_transposed = df_final.T

    # Crear máscara: no colorear la columna 0 (referencia)
    mask = pd.DataFrame(False, index=df_transposed.index, columns=df_transposed.columns)
    mask.iloc[:, 0] = True

    # Dibujar heatmap
    fig1 = plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        df_transposed, annot=False, cmap='RdYlGn', cbar=False,
        linewidths=1, fmt='.0%', vmin=-.8, vmax=.8, mask=mask
    )
    ax.xaxis.tick_top()

    # Anotar valores de la columna 0 (referencia)
    for y, idx in enumerate(df_transposed.index):
        rect = patches.Rectangle((0, y), 1, 1, linewidth=1, edgecolor="w", facecolor='lightgray', alpha=0.5)
        ax.add_patch(rect)
        val = df_transposed.iloc[y, 0]
    
        if "%" in idx.lower():
            texto = f"{val:.0%}"
        elif idx.upper() == "PER":
            texto = f"{val:.0f}"
        else:
            texto = f"{val:.2f}"
    
        ax.text(0.5, y + 0.5, texto, ha='center', va='center', color='black', fontweight='bold', size=10)

    # Anotar diferencias del resto de columnas
    for x in range(1, df_transposed.shape[1]):
        for y, idx in enumerate(df_transposed.index):
            val = df_transposed.iloc[y, x]
            ax.text(x + 0.5, y + 0.5, f"{val:.0%}", ha='center', va='center', color='black')

    ax.yaxis.tick_left()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, weight="bold", rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, weight="bold", rotation=0)

    # Título
    if es_jugador:
        plt.title(f"{referencia} ({s} - {ref_row.teamName.values[0]}) | DISTANCIA vs. Prototipos, por Medida\n", fontsize=10, weight="bold")
    else:
        plt.title("DISTANCIA entre Prototipos, por Medida\n", fontsize=10, weight="bold")

    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    return fig1

def pitch_maker(player,data,cluster,s,color='purple'):
    
    rc = data[(data['playerName_id']==player) & (data.season==s)]
    
    mapper = pd.read_excel('config/POS_mapper.xlsx')
    mapper= mapper[mapper.Position==data[data.playerName_id==player].position.values[0]]
    
    mapper = mapper.drop_duplicates(subset=['map_x','map_y'])
    
    pitch =VerticalPitch(pitch_type='uefa', line_color='grey')
    fig, ax = pitch.draw(figsize=(7, 3))
    pitch.scatter(mapper.iloc[0].map_x,mapper.iloc[0].map_y,s = 400, c=color,ax=ax, zorder=2)
    if mapper.shape[0]>1:
        pitch.scatter(mapper.iloc[1:].map_x,mapper.iloc[1:].map_y,s = 150, 
                      c=color,ax=ax,alpha=.25, zorder=2)
    pv = 20
    if mapper.iloc[0].map_y>68-pv:
        pv = 68 - mapper.iloc[0].map_y
    elif mapper.iloc[0].map_y<20:
        pv = mapper.iloc[0].map_y
    
    l = 10
    
    
    if pv == 20 and rc['position'].values[0]!='GK':
        if rc['position'].values[0]=='AMF':
                rect1 = patches.Rectangle((mapper.iloc[0].map_y,mapper.iloc[0].map_x-l), -pv-10, 20,
                                     linewidth=0, facecolor='lime', alpha=.5)
                
                plt.gca().add_patch(rect1)
                
                rect2 = patches.Rectangle((mapper.iloc[0].map_y,mapper.iloc[0].map_x-l), pv+10, 20,
                                     linewidth=0, facecolor='lime', alpha=.5)
                
                plt.gca().add_patch(rect2)
                
        else:
                rect1 = patches.Rectangle((mapper.iloc[0].map_y,mapper.iloc[0].map_x-l), -pv, 20,
                                     linewidth=0, facecolor='lime', alpha=.5)
                
                plt.gca().add_patch(rect1)
                
                rect2 = patches.Rectangle((mapper.iloc[0].map_y,mapper.iloc[0].map_x-l), pv, 20,
                                     linewidth=0, facecolor='lime', alpha=.5)
                
                plt.gca().add_patch(rect2)
    
    plt.figtext(.5,-0.1, "{}".format(cluster.replace(" ","\n")), size=18,weight='bold',ha = 'center',
                bbox=dict(facecolor=color, edgecolor='black', boxstyle='round', pad=0.2, linewidth=0, alpha=0.2))
    return fig

def scatterplot_plotly(df, df_cols,x_metric, y_metric,how,teams,position_padre):


    # Paleta de colores personalizada
    if how=="rol_desc":
        colores = ['red', 'purple', 'turquoise', 'orange']
        tit = "Rol"
    else:
        colores = ['red', 'yellow', 'green', 'purple']
        tit="Prototipo"
    roles = df[how].unique()

    # Detectar si son porcentajes
    x_is_pct = x_metric.endswith('_pct')
    y_is_pct = y_metric.endswith('_pct')

    # Formatear títulos
    x_title = df_cols[df_cols.medida == x_metric].fancy_name_esp.values[0]
    y_title = df_cols[df_cols.medida == y_metric].fancy_name_esp.values[0]

    # Crear figura
    fig = go.Figure()

    for rol, color in zip(roles, colores):
        df_rol = df[df[how] == rol]

        # Texto de hover personalizado con % si aplica
        hovertemplate = "<b>%{text}</b><br>"
        hovertemplate += f"X: %{{x:.0%}}<br>" if x_is_pct else "X: %{x:.2f}<br>"
        hovertemplate += f"Y: %{{y:.0%}}<extra></extra>" if y_is_pct else "Y: %{y:.2f}<extra></extra>"

        # Scatter normal
        fig.add_trace(go.Scatter(
            x=df_rol[x_metric],
            y=df_rol[y_metric],
            mode='markers',
            marker=dict(color=color, size=10, line=dict(width=1, color='DarkSlateGrey')),
            name=rol,
            text=df_rol['playerName'] + ' - ' + df_rol['teamName'],
            hovertemplate=hovertemplate,
            showlegend=True
        ))

        # Solo etiquetas visibles para jugadores del equipo
        df_equipo = df_rol[df_rol['teamName'] == teams]
        fig.add_trace(go.Scatter(
            x=df_equipo[x_metric],
            y=df_equipo[y_metric],
            mode='text',
            text=df_equipo['playerName'],
            textposition='top center',
            textfont=dict(size=14, color='black', weight='bold'),
            hoverinfo='skip',
            showlegend=False
        ))

    # Actualizar layout con ejes en formato %
    fig.update_layout(
        title={
            'text': f'Dispersión de {position_padre} Destacados por {tit}   |   X: {x_title}, Y: {y_title}',
            'x': 0.5,  # <-- Centrado horizontal
            'xanchor': 'center',
            'font': dict(size=16)
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title=tit,
        height=700
    )

    if x_is_pct:
        fig.update_xaxes(tickformat=".0%")
    if y_is_pct:
        fig.update_yaxes(tickformat=".0%")

    return fig