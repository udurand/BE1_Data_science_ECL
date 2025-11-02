# -*- coding: utf-8 -*-
"""
Ce script implémente l’ensemble du pipeline d’identification d’utilisateurs de Copilote :
- Chargement et exploration des données
- Extraction de caractéristiques comportementales et textuelles (TF-IDF)
- Normalisation et visualisations
- Entraînement de plusieurs modèles de classification
- Sélection du meilleur modèle et export du fichier de prédiction

Auteurs : Groupe Théo forever, Julien Durand, Laurène Cristol, Théo Florence
Cours  : MOD 7.2 – Introduction à la science des données
Date   : Octobre 2025

"""

# ============================================================
# Identification des utilisateurs de Copilote
# MOD 7.2 – Introduction à la science des données
# Séance 1 : Découverte et première analyse des données
# ============================================================


# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import math
import time

from collections import Counter
from scipy.stats import entropy
from IPython.display import display, Markdown

# === Machine Learning & Prétraitement ===
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

start_time = time.time()

print(" Début du processus...")

# ============================================================
# PARTIE I – Chargement et exploration des données
# Objectif : importer les fichiers sources et explorer
# la structure du jeu de données (dimensions, types de variables), 
# réaliser les premières statistiques exploratoires.
# ============================================================


# === 2. Fonction utilitaire pour charger les datasets ===
def read_ds(ds_name: str, path: str = "C:\Julien"):
    """
    Charge le dataset (train ou test) de manière robuste, 
    même si les lignes n'ont pas le même nombre d'actions.
    """
    file_path = f"{path}/{ds_name}.csv"
    with open(file_path, encoding="utf-8") as f:
        # Trouver le nombre maximum de colonnes (actions)
        max_actions = max((len(line.split(",")) for line in f.readlines()))
        f.seek(0)
        # Nommer les colonnes
        _names = ["util", "navigateur"] if "train" in ds_name.lower() else ["navigateur"]
        _names.extend(range(max_actions - len(_names)))
        # Charger en DataFrame
        df = pd.read_csv(file_path, names=_names, dtype=str)
    return df

# === 3. Chargement des données ===
features_train = read_ds("train")
features_test = read_ds("test")

print("Forme des jeux de données :")
print("Train :", features_train.shape)
print("Test  :", features_test.shape)

display(features_train.head(3))

# === 4. Observation du nombre d'actions par session ===
features_train["nb_actions"] = features_train.drop(columns=["util", "navigateur"]).notna().sum(axis=1)

plt.figure(figsize=(10,5))
sns.histplot(features_train["nb_actions"], bins=50, kde=True)
plt.title("Distribution du nombre d'actions par session (train)")
plt.xlabel("Nombre d'actions")
plt.ylabel("Fréquence")
plt.show()

# === 5. Statistiques sur les navigateurs ===
print("\nNombre de sessions par navigateur :")
display(features_train["navigateur"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(y="navigateur", data=features_train, order=features_train["navigateur"].value_counts().index)
plt.title("Répartition des navigateurs utilisés")
plt.xlabel("Nombre de sessions")
plt.ylabel("Navigateur")
plt.show()

# === 6. Fonction utilitaire pour affichage markdown ===
def markdown_table(headNtail=False, use_index=True, title=None, precision=2):
    def _get_value(val): return str(round(val, precision) if isinstance(val, float) else val)
    def _format_row(row):
        row_str = ""
        if use_index: row_str += f"|{str(row.name)}"
        for value in row.values: row_str += f"| {_get_value(value)}"
        return row_str + "|"
    def _get_str(df):
        return "\n".join(df.apply(_format_row, axis=1))
    def _deco(f):
        def _f(*args, **kwargs):
            df = f(*args, **kwargs)
            _str = f"#### {title}\n" if title else ""
            header = ([str(df.index.name)] if use_index else []) + df.columns.astype(str).to_list() 
            _str += f"|{'|'.join(header)}|" + f"\n|{'--|'*len(header)}\n" if header else None
            if headNtail:
                _str += _get_str(df.head())
                _str += "\n|...|...|\n"
                _str += _get_str(df.tail())
            else:
                _str += _get_str(df)
            display(Markdown(_str))
        return _f
    return _deco

# === 7. Navigateur par utilisateur ===
@markdown_table(title="Navigateur par utilisateur (fréquence)", headNtail=True)
def browsers_per_player(df):
    pivot = pd.crosstab(df["util"], df["navigateur"])
    pivot["Total"] = pivot.sum(axis=1)
    return pivot.sort_values("Total", ascending=False)

browsers_per_player(features_train)

# === 8. Distribution de la variable cible (utilisateur) ===
@markdown_table(title="Distribution de la variable cible (utilisateur)", headNtail=True)
def get_Y_stats(df):
    y_counts = df["util"].value_counts().reset_index()
    y_counts.columns = ["utilisateur", "nombre_sessions"]
    y_counts["%"] = (y_counts["nombre_sessions"] / len(df) * 100).round(2)
    return y_counts

get_Y_stats(features_train)

plt.figure(figsize=(12,5))
top_users = features_train["util"].value_counts().head(20)
sns.barplot(x=top_users.index, y=top_users.values)
plt.title("Top 20 utilisateurs les plus actifs")
plt.xticks(rotation=45)
plt.ylabel("Nombre de sessions")
plt.show()

# === 9. Conclusion exploratoire ===
print("""
Premières observations :
- Chaque ligne = une session utilisateur (séquence d’actions)
- 'util' est la cible à prédire (disponible uniquement dans le train)
- 'navigateur' est une variable catégorielle informative
- Le nombre d’actions varie fortement selon les sessions (certaines très longues)
- La plupart des colonnes après la 2e contiennent des NaN (liées à la durée de session)
""")

# === 10. Analyse approfondie : statistiques comportementales ===

def session_stats(row):
    """Calcule plusieurs statistiques comportementales sur une session."""
    actions = [a for a in row[2:] if pd.notna(a)]
    if not actions:
        return pd.Series({
            "total_actions": 0, "n_time_windows": 0, "max_time": 0,
            "unique_actions": 0, "ratio_top_action": 0,
            "mean_actions_per_timewindow": 0
        })

    # Actions et temps
    times = [int(str(a)[1:]) for a in actions if re.match(r"t\d+", str(a))]
    acts = [str(a) for a in actions if not re.match(r"t\d+", str(a))]

    total = len(acts)
    n_time = len(set(times))
    max_t = max(times) if times else 0
    uniq = len(set(acts))
    top = Counter(acts).most_common(1)[0][1] if acts else 0
    ratio = top / total if total else 0
    mean_actions_per_t = total / n_time if n_time > 0 else np.nan

    return pd.Series({
        "total_actions": total,
        "n_time_windows": n_time,
        "max_time": max_t,
        "unique_actions": uniq,
        "ratio_top_action": ratio,
        "mean_actions_per_timewindow": mean_actions_per_t
    })

# Appliquer aux données
stats_df = features_train.apply(session_stats, axis=1)
features_train = pd.concat([features_train, stats_df], axis=1)

# === 11. Statistiques descriptives ===
display(features_train[[
    "total_actions", "n_time_windows", "max_time", 
    "unique_actions", "ratio_top_action", "mean_actions_per_timewindow"
]].describe())

# === 12. Visualisations ===
fig, axes = plt.subplots(2, 3, figsize=(15,8))
sns.histplot(features_train["total_actions"], bins=50, kde=True, ax=axes[0,0])
axes[0,0].set_title("Total d'actions par session")

sns.histplot(features_train["n_time_windows"], bins=30, kde=True, ax=axes[0,1])
axes[0,1].set_title("Nombre de fenêtres temporelles (tXX)")

sns.histplot(features_train["max_time"], bins=30, kde=True, ax=axes[0,2])
axes[0,2].set_title("Durée estimée (max tXX)")

sns.histplot(features_train["unique_actions"], bins=30, kde=True, ax=axes[1,0])
axes[1,0].set_title("Nombre d’actions uniques")

sns.histplot(features_train["ratio_top_action"], bins=30, kde=True, ax=axes[1,1])
axes[1,1].set_title("Ratio de l’action la plus fréquente")

sns.histplot(features_train["mean_actions_per_timewindow"], bins=30, kde=True, ax=axes[1,2])
axes[1,2].set_title("Moyenne d’actions par fenêtre de temps")

plt.tight_layout()
plt.show()

print("Nombre d'utilisateurs distincts :", features_train["util"].nunique())


# ============================================================
# PARTIE II – Construction des caractéristiques (avec TF-IDF)
# Objectif : extraire des caractéristiques numériques à partir des sessions utilisateurs,
# comme la durée, le nombre d'actions, la diversité des pages consultées, etc.
# Puis convertir les séquences de pages visitées en vecteurs numériques
# en utilisant le modèle TF-IDF. Cette méthode mesure l’importance relative
# des n-grammes de pages dans le comportement utilisateur.
# Ces indicateurs serviront ensuite de base à la modélisation.
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer

# === 14. Conversion de la variable dépendante en catégories (train uniquement) ===
features_train["util"] = pd.Categorical(features_train["util"])
features_train["util_code"] = features_train["util"].cat.codes

# === 15. Définition des motifs de parsing ===
pattern_screen = re.compile(r"\((.*?)\)")
pattern_conf_screen = re.compile(r"<(.*?)>")
pattern_chain = re.compile(r"\$(.*?)\$")

def filter_action(value: str):
    """Nettoie une action en supprimant tout ce qui est entre (), <>, $, ou suffixe '1'."""
    for delim in ["(", "<", "$", "1"]:
        if delim in value:
            value = value.split(delim)[0]
    return value

def parse_session_features(row):
    """Analyse une ligne de session et en extrait des features numériques classiques."""
    actions_raw = [str(a) for a in row[2:] if pd.notna(a)]
    times = [int(a[1:]) for a in actions_raw if re.match(r"t\d+", a)]
    acts = [a for a in actions_raw if not re.match(r"t\d+", a)]
    
    total_actions = len(acts)
    n_time_windows = len(set(times))
    max_time = max(times) if times else 0
    unique_actions = len(set([filter_action(a) for a in acts]))
    
    action_counts = Counter([filter_action(a) for a in acts])
    top_action_count = action_counts.most_common(1)[0][1] if action_counts else 0
    ratio_top_action = top_action_count / total_actions if total_actions else 0
    
    screens = [m.group(1) for a in acts for m in [pattern_screen.search(a)] if m]
    conf_screens = [m.group(1) for a in acts for m in [pattern_conf_screen.search(a)] if m]
    chains = [m.group(1) for a in acts for m in [pattern_chain.search(a)] if m]
    
    nb_screens = len(set(screens))
    nb_conf_screens = len(set(conf_screens))
    nb_chain_refs = len(chains)
    nb_modified_actions = sum(a.endswith("1") for a in acts)
    entropy_actions = entropy(list(action_counts.values())) if action_counts else 0
    mean_actions_per_timewindow = total_actions / n_time_windows if n_time_windows > 0 else np.nan
    toasts_count = sum("Affichage d'un toast" in a for a in acts)
    errors_count = sum("Affichage d'une erreur" in a for a in acts)
    
    return pd.Series({
        "total_actions": total_actions,
        "n_time_windows": n_time_windows,
        "max_time": max_time,
        "unique_actions": unique_actions,
        "top_action_count": top_action_count,
        "ratio_top_action": ratio_top_action,
        "nb_screens": nb_screens,
        "nb_conf_screens": nb_conf_screens,
        "nb_chain_refs": nb_chain_refs,
        "nb_modified_actions": nb_modified_actions,
        "entropy_actions": entropy_actions,
        "mean_actions_per_timewindow": mean_actions_per_timewindow,
        "toasts_count": toasts_count,
        "errors_count": errors_count
    })

def build_features_with_tfidf(df_train, df_test, max_tfidf_features=500):
    # 1️ Features classiques
    X_train_num = df_train.apply(parse_session_features, axis=1)
    X_test_num = df_test.apply(parse_session_features, axis=1)
    
    # 2️ Browser One-Hot
    browser_ohe_train = pd.get_dummies(df_train["navigateur"], prefix="browser")
    browser_ohe_test = pd.get_dummies(df_test["navigateur"], prefix="browser")
    X_train_num = pd.concat([X_train_num, browser_ohe_train], axis=1)
    X_test_num = pd.concat([X_test_num, browser_ohe_test], axis=1)
    X_test_num = X_test_num.reindex(columns=X_train_num.columns, fill_value=0)

    # 3 TF-IDF amélioré
    def actions_to_string(row):
        acts = [str(a).lower() for a in row[2:] if pd.notna(a) and not re.match(r"t\d+", str(a))]
        acts = [filter_action(a) for a in acts]
        stop_actions = {"clic", "scroll", "fermeture", "ouverture", "retour"}
        acts = [a for a in acts if a not in stop_actions]
        return " ".join(acts)

    train_texts = df_train.apply(actions_to_string, axis=1)
    test_texts = df_test.apply(actions_to_string, axis=1)

    tfidf = TfidfVectorizer(
        max_features=max_tfidf_features,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        norm='l2',
        lowercase=True
    )
    X_train_tfidf = pd.DataFrame(
        tfidf.fit_transform(train_texts).toarray(),
        columns=[f"tfidf_{i}" for i in range(len(tfidf.get_feature_names_out()))]
    )
    X_test_tfidf = pd.DataFrame(
        tfidf.transform(test_texts).toarray(),
        columns=[f"tfidf_{i}" for i in range(len(tfidf.get_feature_names_out()))]
    )

    # 4️ Concaténer
    X_train_final = pd.concat([X_train_num.reset_index(drop=True), X_train_tfidf], axis=1)
    X_test_final = pd.concat([X_test_num.reset_index(drop=True), X_test_tfidf], axis=1)
    X_train_final["util_code"] = df_train["util_code"].values
    
    return X_train_final, X_test_final, tfidf

# === 17. Application ===
session_features_train, session_features_test, tfidf_vectorizer = build_features_with_tfidf(
    features_train, features_test, max_tfidf_features=500
)

print(" Données train et test prêtes (sans normalisation)")
print("Train shape:", session_features_train.shape)
print("Test shape :", session_features_test.shape)


# ============================================================
# PARTIE III – Statistiques simples sur les variables
# Objectif : analyser les distributions des caractéristiques numériques et textuelles,
# identifier des corrélations, visualiser la structure des données (PCA)
# et comprendre les comportements types selon les utilisateurs.
# ============================================================

# === 17. Séparation des types de variables ===
num_features = [
    "total_actions", "n_time_windows", "max_time", 
    "unique_actions", "top_action_count", "ratio_top_action",
    "nb_screens", "nb_conf_screens", "nb_chain_refs",
    "nb_modified_actions", "entropy_actions", "mean_actions_per_timewindow",
    "toasts_count", "errors_count"
]

cat_features = [col for col in session_features_train.columns if col.startswith("browser_")] + ["util_code"]

# Colonnes TF-IDF
tfidf_features = [c for c in session_features_train.columns if c.startswith("tfidf_")]

print("Variables numériques classiques :", num_features)
print("Variables catégorielles :", cat_features)
print("Variables TF-IDF :", len(tfidf_features))

# === 18. Distribution des variables numériques classiques (brutes) ===
fig, axes = plt.subplots(len(num_features), 2, figsize=(14, len(num_features)*2.5))
for i, feat in enumerate(num_features):
    sns.histplot(session_features_train[feat], bins=50, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f"Distribution: {feat}")
    
    sns.boxplot(x=session_features_train[feat], ax=axes[i, 1])
    axes[i, 1].set_title(f"Boxplot: {feat}")

plt.tight_layout()
plt.show()

# === 19. Visualisation des variables catégorielles ===
for cat in cat_features:
    plt.figure(figsize=(8,4))
    session_features_train[cat].value_counts(normalize=True).plot(kind='bar')
    plt.title(f"Répartition de la variable catégorielle : {cat}")
    plt.xlabel(cat)
    plt.ylabel("Proportion")
    plt.show()

# === 20. Matrice de corrélation pour les variables numériques classiques ===
corr_matrix = session_features_train[num_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de corrélation des variables numériques")
plt.show()

# === 21. Analyse par utilisateur : boxplots sur quelques variables ===
def plot_categories_box(categories, feature="total_actions"):
    """Affiche un boxplot d’un feature pour plusieurs utilisateurs."""
    plt.figure(figsize=(14, 5))
    subset = session_features_train[session_features_train["util_code"].isin(categories)]
    sns.boxplot(x="util_code", y=feature, data=subset)
    plt.title(f"Distribution de '{feature}' pour les utilisateurs {categories}")
    plt.xlabel("Utilisateur (code)")
    plt.ylabel(feature)
    plt.show()

example_users = [0, 15, 152, 199]
for feat in ["total_actions", "unique_actions", "entropy_actions"]:
    plot_categories_box(example_users, feature=feat)

# === 22. Scatter plots multivariés ===
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="total_actions", 
    y="mean_actions_per_timewindow",
    hue="util_code",
    palette="tab20",
    data=session_features_train,
    legend=None
)
plt.title("Relation total_actions vs mean_actions_per_timewindow")
plt.xlabel("Total Actions")
plt.ylabel("Moyenne actions / fenêtre")
plt.show()

# Boucle pour créer un violin plot par variable
vars_to_plot = ["unique_actions", "entropy_actions", "top_action_count"]
for feat in vars_to_plot:
    plt.figure(figsize=(6, 4))
    sns.violinplot(
        data=session_features_train[[feat]],
        inner="quartile"
    )
    plt.title(f"Violin plot – {feat}")
    plt.xlabel(feat)
    plt.show()

# === 23. Statistiques descriptives brutes ===
print(" Statistiques descriptives finales (valeurs brutes) :")
display(session_features_train.describe(include="all"))

# === 24. Visualisation 2D du TF-IDF (PCA) ===
from sklearn.decomposition import PCA

print("\n Réduction de dimension TF-IDF via PCA...")
pca = PCA(n_components=2, random_state=42)
tfidf_pca = pca.fit_transform(session_features_train[tfidf_features])

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=tfidf_pca[:, 0],
    y=tfidf_pca[:, 1],
    hue=session_features_train["util_code"],
    palette="tab20",
    s=25,
    alpha=0.8,
    legend=False
)
plt.title("Projection PCA (2D) des embeddings TF-IDF")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.tight_layout()
plt.show()

# === 25. Corrélation mixte : variables numériques + toutes les TF-IDF ===
print("\n Calcul de la matrice de corrélation mixte (numériques + TF-IDF)...")

# Combinaison : variables numériques + toutes les TF-IDF
num_features_aug = session_features_train[num_features].copy()
all_features_mix = pd.concat([num_features_aug, session_features_train[tfidf_features]], axis=1)

# Calcul de la matrice de corrélation
corr_mixte = all_features_mix.corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_mixte, cmap="coolwarm", center=0, annot=False, square=True,
    cbar_kws={"shrink": 0.8}
)
plt.title("Matrice de corrélation mixte : numériques + TF-IDF")
plt.tight_layout()
plt.show()


# === 26. Visualisation des TF-IDF les plus caractéristiques par utilisateur ===

top_n = 10  # nombre de n-grammes à afficher par utilisateur
selected_users = [0, 15, 152, 199]  # exemples d'utilisateurs

tfidf_names = tfidf_vectorizer.get_feature_names_out()

for user in selected_users:
    user_mask = session_features_train["util_code"] == user
    user_tfidf = session_features_train.loc[user_mask, tfidf_features].mean(axis=0)
    
    top_features_idx = user_tfidf.nlargest(top_n).index
    top_features_values = user_tfidf[top_features_idx].values
    top_features_names = [tfidf_names[int(f.split("_")[1])] for f in top_features_idx]
    
    plt.figure(figsize=(8,4))
    sns.barplot(x=top_features_values, y=top_features_names, palette="viridis")
    plt.title(f"Top {top_n} TF-IDF n-grammes – Utilisateur {user}")
    plt.xlabel("TF-IDF moyen")
    plt.ylabel("N-gram")
    plt.tight_layout()
    plt.show()

# ============================================================
# PARTIE III BIS – Corrélation features / variable cible
# Objectif : analyser les relations statistiques entre les différentes features
# (numériques et textuelles) et la variable cible 'util_code'.
# Cette étape permet d’identifier les variables les plus explicatives
# et d’obtenir une première intuition sur leur pouvoir discriminant.
# ============================================================

print(" Corrélation des features numériques classiques avec la variable 'util_code'")

# === Corrélation 1 : seulement features numériques classiques ===
corr_with_target_num = session_features_train[num_features + ["util_code"]].corr()["util_code"].drop("util_code")
corr_with_target_num = corr_with_target_num.sort_values(key=abs, ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=corr_with_target_num.values, y=corr_with_target_num.index, palette="magma")
plt.title("Corrélation des features numériques classiques avec 'util_code'")
plt.xlabel("Corrélation de Pearson")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\nTop 10 features numériques les plus corrélées avec 'util_code' :")
display(corr_with_target_num.head(10))


# === Corrélation 2 : toutes les features (numériques + TF-IDF) ===
all_features_for_corr = num_features + tfidf_features
corr_with_target_all = session_features_train[all_features_for_corr + ["util_code"]].corr()["util_code"].drop("util_code")
corr_with_target_all = corr_with_target_all.sort_values(key=abs, ascending=False)

# Graphique des top 20 features toutes incluses
top_n = 20
plt.figure(figsize=(10,8))
sns.barplot(
    x=corr_with_target_all.head(top_n).values,
    y=corr_with_target_all.head(top_n).index,
    palette="viridis"
)
plt.title(f"Top {top_n} corrélations avec 'util_code' (numériques + TF-IDF)")
plt.xlabel("Corrélation de Pearson")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print(f"\nTop {top_n} features les plus corrélées avec 'util_code' (numériques + TF-IDF) :")
display(corr_with_target_all.head(top_n))


# === 24. Normalisation robuste finale (numériques + TF-IDF) ===
all_numeric_cols = num_features + tfidf_features

scaler = RobustScaler()
session_features_train[all_numeric_cols] = scaler.fit_transform(session_features_train[all_numeric_cols])
session_features_test[all_numeric_cols] = scaler.transform(session_features_test[all_numeric_cols])

print(" Toutes les features numériques (classiques + TF-IDF) sont normalisées.")

# === 25. Aperçu final ===
display(session_features_train.head(3))
display(session_features_test.head(3))


# ============================================================
# PARTIE IV – Sélection automatique du meilleur modèle de classification
# Objectif : tester plusieurs modèles de classification (Logistic Regression,
# Decision Tree, Random Forest, SVM, MLP) sur le même jeu d'entraînement.
# Les performances sont comparées via des métriques de validation (accuracy, F1-score)
# pour sélectionner le modèle le plus performant.
# ============================================================




# === 2. Séparation features / labels ===
X = session_features_train.drop(columns=["util_code"])
y = session_features_train["util_code"]

# === 4. Split train / validation ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f" Train: {X_train.shape}, Validation: {X_val.shape}")

# === 5. Définition de plusieurs modèles ===
models = {
    "LogisticRegression": LogisticRegression(max_iter=500, solver="lbfgs", multi_class="auto"),
    "DecisionTree": DecisionTreeClassifier(max_depth=30, random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "SVM_rbf": SVC(kernel="rbf", C=2, gamma="scale", random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

# === 6. Entraînement et évaluation ===
results = []
for name, model in models.items():
    print(f"\n🔹 Entraînement du modèle : {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")

    print(f" Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
    results.append({"model": name, "accuracy": acc, "f1_macro": f1})

# === 7. Sélection du meilleur modèle ===
results_df = pd.DataFrame(results).sort_values(by="f1_macro", ascending=False)
print("\n=== Résumé des performances ===")
display(results_df)

best_model_name = results_df.iloc[0]["model"]
best_model = models[best_model_name]
print(f"\n Meilleur modèle sélectionné : {best_model_name}")

# ============================================================
# PARTIE V – Analyse du meilleur modèle
# Objectif : interpréter le modèle retenu à partir des importances de variables,
# des matrices de confusion et des erreurs de prédiction.  
# Cette analyse permet de comprendre comment le modèle prend ses décisions
# et quelles features influencent le plus la classification.
# ============================================================


print(f"\n Évaluation détaillée du modèle : {best_model_name}")
y_val_pred = best_model.predict(X_val)

accuracy  = round(accuracy_score(y_val, y_val_pred), 3)
precision = round(precision_score(y_val, y_val_pred, average="macro", zero_division=0), 3)
recall    = round(recall_score(y_val, y_val_pred, average="macro", zero_division=0), 3)
f1        = round(f1_score(y_val, y_val_pred, average="macro", zero_division=0), 3)

print(f"\n=== Résultats ===")
print(f"Accuracy : {accuracy}")
print(f"Precision (macro): {precision}")
print(f"Recall (macro)   : {recall}")
print(f"F1-score (macro) : {f1}")

print("\n=== Rapport de classification ===")
print(classification_report(y_val, y_val_pred, zero_division=0))

# === Matrice de confusion (Top 20 classes) ===
top_classes = y_val.value_counts().head(20).index
mask = y_val.isin(top_classes)
cm = confusion_matrix(y_val[mask], y_val_pred[mask], labels=top_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="crest",
            xticklabels=top_classes, yticklabels=top_classes)
plt.title(f"Matrice de confusion (Top 20 classes) – {best_model_name}")
plt.xlabel("Prédictions")
plt.ylabel("Valeurs réelles")
plt.tight_layout()
plt.show()

# === Importance des variables ===
print("\n=== Importance des variables ===")
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
else:
    print("Calcul de l’importance par permutation...")
    perm_importance = permutation_importance(best_model, X_val, y_val, n_repeats=5, random_state=42)
    importances = pd.Series(perm_importance.importances_mean, index=X.columns)

importances = importances.sort_values(ascending=False)
top_features = importances.head(10)

plt.figure(figsize=(8, 5))
sns.barplot(y=top_features.index, x=top_features.values, palette="viridis")
plt.title(f"Top 10 des variables les plus importantes ({best_model_name})")
plt.xlabel("Importance relative")
plt.tight_layout()
plt.show()

# ============================================================
# PARTIE VI – Réentraînement complet et génération de la soumission
# Objectif : réentraîner le modèle sélectionné sur l’ensemble des données disponibles
# pour maximiser ses performances avant de produire les prédictions finales.
# Les résultats sont ensuite sauvegardés sous forme de fichier CSV pour soumission.
# ============================================================

print("\n Réentraînement complet sur toutes les données...")
best_model.fit(X, y)

print(" Prédiction sur le jeu de test...")
preds_test_codes = best_model.predict(session_features_test)
preds_test = pd.Categorical.from_codes(preds_test_codes, features_train["util"].cat.categories)

df_subm = pd.DataFrame(preds_test, columns=["prediction"])
df_subm.index = df_subm.index + 1
df_subm = df_subm.rename_axis("RowId")  

df_subm.to_csv("submission.csv", index=True)

print("Fichier 'submission.csv' créé avec succès !")
print(f"\n Temps total d'exécution : {time.time() - start_time:.1f} secondes.")

# Fin du script – toutes les étapes d’analyse et de modélisation sont terminées.
# Le fichier submission.csv contient la prédiction finale pour le jeu de test.
