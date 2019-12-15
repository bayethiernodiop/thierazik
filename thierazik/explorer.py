import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
import numpy as np
from itertools import cycle
from matplotlib import collections as collections
from matplotlib.patches import Rectangle

"""
@La distribution empirique d'une variable:
    c'est l’ensemble des valeurs (ou modalités) prises par cette variable, 
    ainsi que leurs effectifs associés. 
On trouve aussi une autre version : 
    l’ensemble des valeurs (ou modalités) prises par cette variable, 
    ainsi que leurs fréquences associées. 
"""
def uv_DEQL(x:pd.DataFrame):
    effectifs = x.iloc[:,0].value_counts()
    
    modalites = effectifs.index
    col_names= x.columns
    tab = pd.DataFrame(modalites, columns = col_names) # création du tableau à partir des modalités
    tab["Count"] = effectifs.values
    
    tab["frequency"] = tab["Count"] / len(x)
    
    tab = tab.sort_values(by='Count',ascending=False) # tri des valeurs de la variable X (croissant)
    
    tab["frequency_count"] = tab["frequency"].cumsum()
    
    return tab.transpose()

def uv_showDEQL(x:pd.DataFrame):
    effectifs = x.iloc[:,0].value_counts()
    
    effectifs.plot(kind='bar')
    plt.show()
    

def bv_cql_qt(x:pd.DataFrame,y:pd.DataFrame):

    moyenne_y = y.mean()

    classes = []

    for classe in x.unique():

        yi_classe = y[x==classe]

        classes.append({'ni': len(yi_classe),

                        'moyenne_classe': yi_classe.mean()})

    SCT = sum([(yj-moyenne_y)**2 for yj in y])

    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])

    return SCE/SCT

def bv_cql_ql(x:pd.DataFrame,y:pd.DataFrame):
    tmpdata = x.join(y)          
     
    c = tmpdata.pivot_table(index=x,columns=y,aggfunc=len)
    cont = c.copy()
    tx = x.iloc[:,0].value_counts()

    ty = y.iloc[:,0].value_counts()


    cont.loc[:,"Total"] = tx

    cont.loc["total",:] = ty

    cont.loc["total","Total"] = len(tmpdata)
    
    tx = pd.DataFrame(tx)

    ty = pd.DataFrame(ty)
    tx.columns = ["mmnom"]

    ty.columns = ["mmnom"]
    
    n = len(tmpdata)
    indep = tx.dot(ty.T) / n


    c = c.fillna(0) # on remplace les valeurs nulles par des 0

    mesure = (c-indep)**2/indep

    xi_n = mesure.sum().sum()

    sns.heatmap(mesure/xi_n,annot=c)

    plt.show()


    

    return cont
    

def uv_replace_outlier(x:pd.DataFrame):#,repl):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    #x[(x < fence_low) & (x > fence_high)]=repl
    x.plot(kind='box',vert=False)
    plt.show()
    return x




def fillingfactor(x:pd.DataFrame):
    missing_df = x.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['filling_factor'] = (x.shape[0] - missing_df['missing_count']) / x.shape[0] * 100
    return missing_df.sort_values('filling_factor').reset_index(drop = True)

def vartype(x:pd.DataFrame):
    tab_info=pd.DataFrame(x.dtypes).T.rename(index={0:'column type'})
    tab_info=tab_info.append(pd.DataFrame(x.isnull().sum()).T.rename(index={0:'null values'}))
    tab_info=tab_info.append(pd.DataFrame((x.isnull().sum()/x.shape[0])*100).T.rename(index={0:'null values (%)'}))
    return tab_info
    

def showfillingfactordist(x:pd.DataFrame,size=(20,5)):
    missing_df = fillingfactor(x)
    y_axis = missing_df['filling_factor'] 
    x_label = missing_df['column_name']
    x_axis = missing_df.index
    yy_axis = missing_df['missing_count']

    fig = plt.figure(figsize=size)
    plt.xticks(rotation=90, fontsize = 16)
    plt.yticks(fontsize = 16)

    plt.axhline(20, linewidth=2, color = '#ff0000')
    plt.text(missing_df.shape[0]//2, 21, '20%',fontsize = 25,color="#ffffff",weight='bold',
             bbox=dict(boxstyle="round",facecolor='#ff0000',edgecolor='#ff0000'))
    plt.axhline(40, linewidth=2, color = '#ff6600')
    plt.text(missing_df.shape[0]//2, 41, '40%',fontsize = 25, color="#ffffff",weight='bold', 
             bbox=dict(boxstyle="round",facecolor='#ff6600',edgecolor='#ff6600'))
    plt.axhline(60, linewidth=2, color = '#0066ff')
    plt.text(missing_df.shape[0]//2, 61, '60%',fontsize = 25,color="#ffffff",weight='bold',
             bbox=dict(boxstyle="round",facecolor='#0066ff',edgecolor='#0066ff'))
    plt.axhline(80, linewidth=2, color = '#009900')
    plt.text(missing_df.shape[0]//2, 81, '80%',fontsize = 25,color="#ffffff",weight='bold',
             bbox=dict(boxstyle="round",facecolor='#009900',edgecolor='#009900'))
   
    plt.xticks(x_axis, x_label, fontsize = 16 )
    plt.ylabel('Filling factor (%)',  fontsize = 16)
    plt.bar(x_axis, y_axis);
    plt.show()
    
    # show missing factor
    fig = plt.figure(figsize=size)
    plt.xticks(rotation=90, fontsize = 16)
    plt.yticks(fontsize = 16)
      
    plt.xticks(x_axis, x_label, fontsize = 16 )
    plt.ylabel('Missing counts ',  fontsize = 16)
    
    plt.bar(x_axis, yy_axis);
    plt.show()
    return missing_df.set_index('column_name').transpose()
    

def centrage(x:pd.DataFrame):
    return x-x.mean()

def reduction(x:pd.DataFrame,d):
    return centrage(x)/x.std(ddof=d)


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None,fsize=(15,15)):

    for d1, d2 in axis_ranks: 

        if d2 < n_comp:


            # initialisation de la figure

            fig, ax = plt.subplots(figsize=fsize)


            # détermination des limites du graphique

            if lims is not None :

                xmin, xmax, ymin, ymax = lims

            elif pcs.shape[1] < 30 :

                xmin, xmax, ymin, ymax = -1, 1, -1, 1

            else :

                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])


            # affichage des flèches

            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité

            if pcs.shape[1] < 30 :

                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),

                   pcs[d1,:], pcs[d2,:], 

                   angles='xy', scale_units='xy', scale=1, color="grey")

                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)

            else:

                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]

                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            

            # affichage des noms des variables  

            if labels is not None:  

                for i,(x, y) in enumerate(pcs[[d1,d2]].T):

                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :

                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.9)

            

            # affichage du cercle

            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')

            plt.gca().add_artist(circle)


            # définition des limites du graphique

            plt.xlim(xmin, xmax)

            plt.ylim(ymin, ymax)

        

            # affichage des lignes horizontales et verticales

            plt.plot([-1, 1], [0, 0], color='grey', ls='--')

            plt.plot([0, 0], [-1, 1], color='grey', ls='--')


            # nom des axes, avec le pourcentage d'inertie expliqué

            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))

            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))


            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))

            plt.show()

        

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):

    for d1,d2 in axis_ranks:

        if d2 < n_comp:

 

            # initialisation de la figure       

            fig = plt.figure(figsize=(7,6))

        

            # affichage des points

            if illustrative_var is None:

                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)

            else:

                illustrative_var = np.array(illustrative_var)

                for value in np.unique(illustrative_var):

                    selected = np.where(illustrative_var == value)

                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)

                plt.legend()


            # affichage des labels des points

            if labels is not None:

                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):

                    plt.text(x, y, labels[i],

                              fontsize='14', ha='center',va='center') 

                

            # détermination des limites du graphique

            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1

            plt.xlim([-boundary,boundary])

            plt.ylim([-boundary,boundary])

        

            # affichage des lignes horizontales et verticales

            plt.plot([-100, 100], [0, 0], color='grey', ls='--')

            plt.plot([0, 0], [-100, 100], color='grey', ls='--')


            # nom des axes, avec le pourcentage d'inertie expliqué

            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))

            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))


            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))

            plt.show(block=False)


def display_scree_plot(pca):

    scree = pca.explained_variance_ratio_*100

    plt.bar(np.arange(len(scree))+1, scree)

    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')

    plt.xlabel("rang de l'axe d'inertie")

    plt.ylabel("pourcentage d'inertie")

    plt.title("Eboulis des valeurs propres")

    plt.show(block=False)
    
 


def missmap(df, ax=None, colors=None, aspect=4, sort='descending',
            title=None, **kwargs):
    """
    Plot the missing values of df.

    Parameters
    ----------
    df : pandas DataFrame
    ax : matplotlib axes
        if None then a new figure and axes will be created
    colors : dict
        dict with {True: c1, False: c2} where the values are
        matplotlib colors.
    aspect : int
        the width to height ratio for each rectangle.
    sort : one of {'descending', 'ascending', None}
    title : str
    kwargs : dict
        matplotlib.axes.bar kwargs

    Returns
    -------
    ax : matplotlib axes

    """
    plt.figure(figsize=(25,25))
    if ax is None:
        fig, ax = plt.subplots()
    
    # setup the axes
    dfn = pd.isnull(df)

    if sort in ('ascending', 'descending'):
        counts = dfn.sum()
        sort_dict = {'ascending': True, 'descending': False}
        counts = counts.sort_values(ascending=sort_dict[sort])
        dfn = dfn[counts.index]

    # Up to here
    ny = len(df)
    nx = len(df.columns)
    # each column is a stacked bar made up of ny patches.
    xgrid = np.tile(np.arange(nx), (ny, 1)).T
    ygrid = np.tile(np.arange(ny), (nx, 1))
    # xys is the lower left corner of each patch
    xys = (zip(x, y) for x, y in zip(xgrid, ygrid))

    if colors is None:
        colors = {True: '#EAF205', False: 'k'}

    widths = cycle([aspect])
    heights = cycle([1])

    for xy, width, height, col in zip(xys, widths, heights, dfn.columns):
        color_array = dfn[col].map(colors)

        rects = [Rectangle(xyc, width, height, **kwargs)
                 for xyc, c in zip(xy, color_array)]

        p_coll = collections.PatchCollection(rects, color=color_array,
                                             edgecolor=color_array, **kwargs)
        ax.add_collection(p_coll, autolim=False)

    # post plot aesthetics
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    ax.set_xticks(.5 + np.arange(nx))  # center the ticks
    ax.set_xticklabels(dfn.columns)
    for t in ax.get_xticklabels():
        t.set_rotation(90)

    # remove tick lines
    ax.tick_params(axis='both', which='both', bottom='off', left='off',
                   labelleft='off')
    ax.grid(False)

    if title:
        ax.set_title(title)
    return ax