# 04/18 Création Morgan SCAO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from time import time

# Méthodes d'import export
import pickle
import os
from sklearn.externals import joblib


# Répertoires locaux
#CT_DIR_SAVE = '../autotag/save/'
CT_DIR_SAVE = ''

# Méthode de suppression de colonne
def DropCol(p_df, p_col):
    if p_col in p_df.columns:
        p_df = p_df.drop([p_col], axis=1)
    return p_df

def save_obj(obj, name):
    fn = CT_DIR_SAVE + name + '.pkl'
    try:
        os.remove(fn)
    except OSError:
        pass
    with open(fn, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print(fn, 'saved')
def load_obj(name):
    print(name, 'loaded')
    with open(CT_DIR_SAVE + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def save_sklearn_obj(obj, name):
    fn = CT_DIR_SAVE + name + '.pkl'
    try:
        os.remove(fn)
    except OSError:
        pass
    joblib.dump(obj, fn)
    print(fn, 'saved')
def load_sklearn_obj(name):
    print(name, 'loaded')
    return joblib.load(CT_DIR_SAVE + name + '.pkl')


import nltk
from nltk.corpus import stopwords
mystops = set(stopwords.words("english"))

from bs4 import BeautifulSoup
import re
# Transformation de la feature 'Body'
def body_to_words(value):
    # 1. Remove HTML
    review_text = BeautifulSoup(value, "lxml").get_text() 
    # 2. Remove special characters
    letters_only = re.sub("[^a-zA-Z0-9#+]", " ", review_text) 
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in mystops]
    # 5. Stemming
    porter = nltk.PorterStemmer()
    meaningful_words = [porter.stem(w) for w in meaningful_words]
    # 6. Join the words back into one string separated by space, and return the result.
    return( " ".join( meaningful_words ))   

# Transformation des tags, on va pouvoir les séparer de leur balise
def tag_to_words(value):
    # On sépare les tags des balises
    words = value.replace('><', ' ')
    words = words.replace('>', '')
    words = words.replace('<', '')
    # On converti en minuscule et on en fait un tableau
    tab_words = words.lower().split()                             
    # On enlève les stopwords
    meaningful_words = [w for w in tab_words if not w in mystops]
    # On retourne une chaine de mots
    return(" ".join(tab_words))

# Méthode d'affichage des mots les plus utilisés
def word_occurrence(feature, top=20):
    count_keywords = dict()
    for row in feature:
        for word in row.split():
            if word in count_keywords:
                count_keywords[word] += 1
            else:
                count_keywords[word] = 1
    liste = []
    for k, v in count_keywords.items():
        liste.append([k,v])

    liste = sorted(liste, key=lambda x:x[1], reverse=True)
    print("Nombre de mots total : {} (pour {} lignes)".format(len(liste), feature.count()))

    plt.rc('font', weight='normal')
    fig, ax = plt.subplots(figsize=(15, 8))
    y_axis = [i[1] for i in liste[:top]]
    x_axis = [k for k,i in enumerate(liste[:top])]
    x_label = [i[0] for i in liste[:top]]
    plt.xticks(fontsize=15, rotation=90)
    plt.yticks(fontsize=13)
    plt.xticks(x_axis, x_label)
    plt.ylabel("Nb. of occurences", fontsize=18, labelpad=10)
    ax.bar(x_axis, y_axis, align='center', color='c')
    ax = plt.gca()
#    ax.invert_yaxis()

    plt.title("Words occurence : top"+str(top), bbox={'facecolor':'k', 'pad':5}, color='w', fontsize = 25)
    plt.show()
    
    # On renvoie la liste triée par nombre d'occurrences
    return liste

# Méthode pour récupérer les meilleurs tags en fonction du pourcentage de la prédiction
def get_best_tags(clf, X, lb, n_tags=5, b_save=False, b_Dir=''):
    decfun = []
    if hasattr(clf, 'decision_function'):
        decfun = clf.decision_function(X)
    elif hasattr(clf, 'predict_proba'):
        decfun = clf.predict_proba(X)
    else:
        return None

    best_tags = np.argsort(decfun)[:, :-(n_tags+1): -1]

    if b_save:
        # Sauvegarde du modèle pour pouvoir le rejouer dans l'API
        save_obj(clf, b_Dir + 'Classifier')
        save_obj(lb, b_Dir + 'MultiLabelBinarizer')

    return lb.classes_[best_tags]


	
def fillBlanks(df):
    print('*** fillBlanks ***')

    # On enlève les colonnes dont toutes les valeurs sont nulles
    df.dropna(axis=1, how='all', inplace=True)

    for i in df.columns:
        if df[i].isnull().sum() == 0:
            continue
    
        if df[i].dtype == object:
            # On remplace par la valeur la plus courante
            v = df[i].mode().values[0]
            df[i] = df[i].fillna(v)
            print('\tValeurs manquantes de', i, 'remplacés par', v)
        else:
            # On remplace par la médiane (pour diminuer l'impact des outliers)
            med = int(np.median(df[i].notnull()))
            df[i] = df[i].fillna(med)
            print('\tValeurs manquantes de', i, 'remplacés par', med)

    return df

# Transformation de la feature Date en age
def anneeToAge(value):
    if value > 1800:
        return dt.datetime.now().year - value
    return value

# Transformation du code NAF de niveau 5 en code NAF de niveau 1
def nafToNaf1(value):
    try:
        if len(value) < 2: return ''
        n = int(value[:2])
        if n<4: return 'A'
        if n<10: return 'B'
    except:
        pass
    return ''

# Définition de la target : procol ou pas procol
def setTarget(value):
    return (len(value) == 0)

def showMissingValues(df):
    print('*** showMissingValues ***')
    tab_info = pd.DataFrame(df.dtypes).T.rename(index={0:'Type'})
    tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'Valeurs manquantes (nb)'}))
    tab_info = tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0:'Valeurs manquantes (%)'}))
    display(tab_info)
    # Plus graphique
    x1=df.isnull().sum().values
    x2=df.notnull().sum().values
    index = np.arange(len(x1))
    fig, ax = plt.subplots(figsize=(16, 22))
    plt.barh(index, x1+x2, color='b')
    plt.barh(index, x1, color='r')
    ax.set_yticks(index)
    ax.set_yticklabels(df.columns.values.tolist())
    plt.title("Proportion de valeurs absentes")
    plt.show()

def CheckOutliers(p_df, bModif=False):
    print('*** CheckOutliers ***')
    for i in p_df.columns:
        if p_df[i].dtype == 'object': continue

        # Etude sur un échantillon représentatif (les valeurs positives)
        df = p_df[p_df[i] > 0]
        if df.shape[0] == 0:
            continue

        # Outliers
        q75, q25 = np.percentile(df[i], [75 ,25])
        iqr = q75 - q25

        valmax = q75 + (iqr * 10) # Je prends une valeur haute car il sinon on sort trop de lignes
        nbOutliers = p_df[p_df[i]>valmax].shape[0]
        if nbOutliers > 0:
            print("%i valeurs supérieures à %f pour %s (max=%f)" % (nbOutliers, valmax, i, np.max(p_df[i])))
            if bModif:
                outliers = p_df[i][ p_df[i]>valmax ]
                p_df.loc[outliers.index, i] = valmax
                #p_df = p_df[(p_df[i] > nbmax) == False]
    return p_df

def complete_and_clean(dfPred, bBilan=False, cj=0, verbose=True):
    print('*** complete_and_clean ***')
    dropcols = []

    # On force les features catégorielles
    #dfPred ["ii_ACTIVNAT"] = dfPred ["ii_ACTIVNAT"].astype(np.str)
    # On fait des transfo
    #dfPred['ii_AGE'] = dfPred['ii_DAPET'].apply(anneeToAge)
    #dropcols.append('ii_DAPET')

    if verbose:
        # Description avant nettoyage
        display(df.describe())
        display(df.select_dtypes(exclude=[np.number]).describe())

    if verbose:
        # Quantité des différents éléments
        print('\n')
        #display(pd.DataFrame([{'Entreprises': df['siren'].nunique(),    
        #        'NAF': df['ii_APE_ENT'].nunique(),
        #        }], columns = ['Entreprises', 
        #                        'NAF', 
        #            ], index = ['Quantité']))
                
    # Suppression de colonnes qui vont bien
    for col in dropcols:
        df = DropCol(df, col)

    if verbose:
        # Analyse des valeurs manquantes
        showMissingValues(df)

    # Bornage des outliers
    df = CheckOutliers(df, True)
    
    # Remplissage des champs vides
    df = fillBlanks(df)

    # Cible
    #df['target'] = df['procol'].apply(setTarget)

    if verbose:
        print('\n')
        #print(df[df['target']==1].shape[0], 'SIREN actifs il y a 12 mois')
        #print('\t', df[(df['target']==1) & (df['indiScoreMoins1']>6)].shape[0], 'TP (True Positifs)')
    
    return df
	
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepareData(p_df, p_dropcols, p_strat=None, bSave=True):
    # La target
    y = p_df['target']

    # On enlève la target des data bien sûr
    X = DropCol(p_df, 'target')
    # Ainsi que les colonnes non désirées
    for col in p_dropcols:
        X = DropCol(X, col)

    scalingDFcols = []
    categDFcols = []
    for col in X.columns:
        if (X[col].dtype == np.object):
            categDFcols.append(col)
        else:
            scalingDFcols.append(col)
    print('Numérique :\n\t', scalingDFcols)
    print('Catégories :\n\t', categDFcols)

    for col in categDFcols:
        lst = load_obj('column_' + col)
        X[col] = X[col].astype('category', categories=lst)

    categDF = X[categDFcols]
    scalingDF = X[scalingDFcols]
    
    # Binarisation en dummies pour garder la maitrise des noms des colonnes
    categDF_encoded = pd.get_dummies(categDF)
    print('Après binarisation les catégories prennent', categDF_encoded.shape[1], 'dimensions.')

    # Concaténation
    x_final = pd.concat([scalingDF, categDF_encoded], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_final, y, test_size = 0.2, random_state=0, stratify=p_strat)
    
    # Seules les xnum premières colonnes sont numériques
    xnum = scalingDF.shape[1]
    x_train_numerical = x_train.iloc[:, 0:xnum]
    x_test_numerical = x_test.iloc[:, 0:xnum]
    x_final_numerical = x_final.iloc[:, 0:xnum]

    # Création d'un scaler pour les valeurs numériques 
    scaler = StandardScaler()
    # Qu'on entraine avec le training set
    scaler.fit(x_train_numerical) 

    if bSave:
        # Sauvegarde
        save_obj(scalingDFcols, 'model_columnsScale'+str(x_final.shape))
        save_obj(categDFcols, 'model_columnsCateg'+str(x_final.shape))
        save_obj(x_final.columns, 'model_columnsAll'+str(x_final.shape))
        save_sklearn_obj(scaler, 'model_scaler'+str(x_final.shape))
    
    x_train_numerical = scaler.transform(x_train_numerical)
    x_test_numerical = scaler.transform(x_test_numerical)
    x_final_numerical = scaler.transform(x_final_numerical)

    x_train = x_train.copy()
    x_test = x_test.copy()
    x_final = x_final.copy()
    x_train.loc[:, 0:xnum] = x_train_numerical
    x_test.loc[:, 0:xnum] = x_test_numerical
    x_final.loc[:, 0:xnum] = x_final_numerical

    print('x_train :', x_train.shape)
    return x_train, x_test, y_train, y_test, x_final


# Matrice de confusion
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_score, recall_score
np.set_printoptions(precision=2)
class_names = ['False', 'True']

# Affichage de matrice de confusion
def show_confusion_matrix(y_reel, y_pred_proba, y_pred=[]):
    if len(y_pred)==0:
        y_pred=y_pred_proba
    # Compute confusion matrix - TOUTE LES DONNEES
    cnf_matrix = confusion_matrix(y_reel, y_pred)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_reel, y_pred_proba)
    tn, fp, fn, tp = cnf_matrix.ravel()
    # Aire sous la courbe
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print ("\tAUC = %.3f" % roc_auc)
    print ("\tSpécificité = %.3f" % (tn/(tn+fp)))
    print ("\tPrecision = %.3f" % precision_score(y_reel, y_pred))
    print ("\tRecall = %.3f" % recall_score(y_reel, y_pred))

    # Plot normalized & non-normalized confusion matrix
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Matrice brute')
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Matrice normalisée')
    plt.show()
    return false_positive_rate,true_positive_rate, roc_auc

def plot_confusion_matrix(cm, classes=['False', 'True'], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # Possibilité de normalisation
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')
    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

# Réduction dimensionnelle
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Permet de récupérer le nombre de composants nécessaires à un pourcentage de variance donné
def select_n_components(var_ratio, goal_var: float) -> int:
    total_variance = 0.0
    n_components = 0
    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break
    # Return the number of components
    return n_components
def doReduc(reduc, X, lbl, std=False, pct=0.90):
    t0 = time()
    if std:
        # Standardize the feature matrix
        X = StandardScaler().fit_transform(X)
    X_red = reduc.fit_transform(X)
    best_n_components = select_n_components(reduc.explained_variance_ratio_, pct)
    print("%s done in %0.3fs." % (lbl, time() - t0))
    print('Nombre de dimensions original :', X.shape[1])
    print(best_n_components, ' dimensions pour', int(pct*100), '% de variance')
    graphExplainedVarianceRation(reduc.explained_variance_ratio_)
    return reduc, X_red, best_n_components
    
def testPCA(X, std=False, pct=0.90):
    # Create and run PCA
    reduc = PCA(n_components=pct, whiten=True)
    return doReduc(reduc, X, 'PCA', std, pct)

# Spécial sparses matrices
def testTSVD_sparse(X_sparse, std=False, pct=0.90):
    # Create and run an TSVD with one less than number of features
    reduc = TruncatedSVD(n_components=X_sparse.shape[1]-1)
    return doReduc(reduc, X_sparse, 'TSVD', std, pct)
def testTSVD(X, std=False, pct=0.90):
    # Make sparse matrix
    X_sparse = csr_matrix(X)
    return testTSVD_sparse(X_sparse, std, pct)

def testLDA(X, y, pct=0.90):
    reduc = LinearDiscriminantAnalysis(n_components=None)
    t0 = time()
    X_red = reduc.fit_transform(X, y)
    best_n_components = select_n_components(reduc.explained_variance_ratio_, pct)
    print("LDA done in %0.3fs." % (time() - t0))
    print('Nombre de dimensions original :', X.shape[1])
    print(best_n_components, ' dimensions pour', int(pct*100), '% de variance')
    graphExplainedVarianceRation(reduc.explained_variance_ratio_)
    return reduc, X_red, best_n_components


def graphExplainedVarianceRation(evr):
    nbvar = len(evr) +1
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.bar(range(1,nbvar), evr, alpha = 0.5, align = 'center', label = 'individual explained variance')
    plt.step(range(1,nbvar), np.cumsum(evr), where = 'mid', label = 'cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc = 'best')
    plt.title('Variance cumulée', fontsize=18)
    plt.show()

def displayPCA(p_df, p_color):
    X_scaled = p_df
    #try:
    #    X_scaled = StandardScaler().fit_transform(p_df.fillna(0))
    #except:
    #    pass
    
    pca = PCA(n_components=None)
    pca.fit(X_scaled)

    graphExplainedVarianceRation(pca.explained_variance_ratio_)

    print("Deux composantes nous permettent d'expliquer %.2f pourcent de la variance" % (np.cumsum(pca.explained_variance_ratio_[:2])[1]*100))
    print('\n')
    
    # projeter X sur les composantes principales
    X_projected = pca.transform(X_scaled)
    # afficher chaque observation
    fig = plt.figure(figsize=(16, 10))
    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=p_color)
    plt.xlim([-5.5, 5.5])
    plt.ylim([-4, 4])
    plt.colorbar()
    plt.title('Projection sur les composantes principales', fontsize=18)
    plt.show()

    # S'il y a trop de feature on n'affiche pas ce dernier graphe qui sera illisible
    if nbvar > 15: return
    print('\n')
    pcs = pca.components_
    fig = plt.figure(figsize=(16, 10))
    for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
        # Afficher un segment de l'origine au point (x, y)
        plt.plot([0, x], [0, y], color='k')
        # Afficher le nom (data.columns[i]) de la performance
        plt.text(x, y, p_df.columns[i], fontsize='18')
    # Afficher une ligne horizontale y=0
    plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')
    # Afficher une ligne verticale x=0
    plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')
    plt.xlim([-0.7, 0.7])
    plt.ylim([-0.7, 0.7])
    plt.title('Contribution de chaque variable aux composantes principales', fontsize=18)
    plt.show()
    return

def PieCategorie(p_df, p_col, y='target'):
    print("Répartition selon", p_col)
    #dftmp2 = p_df[pd.notnull(p_df[p_col])][p_df[p_col] != '0']
    #dftmp2 = dftmp2[dftmp2[p_col] != 'unknown']
    dftmp2 = p_df[pd.notnull(p_df[p_col])]
    # Parcours des différentes valeurs de la catégorie
    for grp in dftmp2[p_col].unique():
        # Réduction de la matrice
        dftmp = dftmp2[[p_col, y]][dftmp2[p_col] == grp]
        nb = dftmp.shape[0]
        dftmp = dftmp.groupby([p_col, y])
        dftmp = dftmp.size().reset_index(name='counts')
        # Affichage
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.pie(dftmp.counts, labels=dftmp[y])
        ax.set_title(p_col + " = " + (str)(grp) + " : " + (str)(nb) + " SIREN")
        #plt.legend()
    plt.show()

def GraphWithSize(XX, yy, titre):
    fig = plt.figure(figsize=(8, 4))
    ax = plt.subplot()
    sizes = {} # clé : coordonnées ; valeur : nombre de points à ces coordonnées
    for (yt, yp) in zip(list(XX), list(yy)):
        if (yt, yp) in sizes:
            sizes[(yt, yp)] += 1
        else:
            sizes[(yt, yp)] = 1
    keys = sizes.keys()
    ax.scatter([k[0] for k in keys], # score en abscisse
    [k[1] for k in keys], # grade en ordonnée
    s=[(sizes[k])/100 for k in keys]) # taille du marqueur

    ax.set_title(titre)
    plt.show()
    
#CORRELATION
def correlation(p_df, bDetails=False):
    corr1 = p_df.corr()
    #print(corr1)

    # Masquage de la diagonale
    mask = np.ones(corr1.columns.size) - np.eye(corr1.columns.size)
    corrMasqued = mask * corr1

    fig = plt.subplots(figsize=(16,10))
    # ax.set_title('Corrélation des features')
    # heatmap = ax.pcolor(corrMasqued, cmap=plt.cm.Greens)
    # plt.show()
    if bDetails:
        sns.heatmap(corr1, annot = True, fmt = ".2f", cbar = True)
    else:
        sns.heatmap(corr1)
    plt.show()

    # Analyse de la matrice de corrélation - Colonnes corrélées
    corrCols = []
    for col in corrMasqued.columns.values:
        # Si la feature a déjà été traitée on passe
        if np.in1d([col], corrCols):
            continue
        
        # Récupération des features fortement corrélées 
        corr = corrMasqued[abs(corrMasqued[col]) > 0.5].index
        corrCols = np.union1d(corrCols, corr)
        
        if corr.shape[0] > 0:
            print("%s corrélé à : %s" % (col, corr.tolist()))

import seaborn as sns
def GraphSeaborn(p_df, p_cat):
    cols = []
    count = 0
    p_df = p_df[pd.notnull(p_df[p_cat])]
    for col in p_df.columns:
        if p_df[col].dtype == 'object': continue
        count += 1
        cols.append(col)
        if count > 0 and count % 4 == 0:
            cols.append(p_cat)
            #print (cols)
            # Si toutes les valeurs sont nulles on enlève la ligne
            #d = p_df[sum(p_df[cols], axis=1) > 0]
            #print(d.shape)
            #sns.pairplot(d[cols], hue=p_cat, size=2.5);
            sns.pairplot(p_df[cols], hue=p_cat, size=2.5);
            cols.clear()
            plt.show()

    if len(cols) > 0:
        cols.append(p_cat)
        #print (cols)
        sns.pairplot(p_df[cols], hue=p_cat, size=2.5);
        plt.show()

def doBoxPlot(p_df, y_col):
    for col in p_df.columns:
        if p_df[col].dtype == 'object': continue
        if y_col == col:
            continue
        # Etude sur un échantillon représentatif (valeurs positives)
        df = p_df[p_df[col]>0]
        #print(df.shape, col)
        #if df.shape[0] == 0: continue

        fig, ax = plt.subplots(figsize=(6,4))
        fig = df.boxplot(col, y_col, ax=ax, grid=False)
        plt.suptitle("")
        plt.show()

def graphWithPlot(p_df, XX, yy, titre):
    dfgroup = p_df[[XX, yy]].groupby([XX]).mean()
    ax = dfgroup.plot.bar(figsize=(11, 6))
    ax.set_xlabel(XX)
    ax.set_ylabel(yy)

