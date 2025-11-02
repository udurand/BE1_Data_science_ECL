# BE1_Data_science_ECL : Identification des utilisateurs de Copilote

**MOD 7.2 – Introduction à la science des données**
BE séances 1, 2, 3 – Octobre 2025

---

## Description

Ce projet vise à identifier les utilisateurs d'une application ("Copilote") à partir de leurs sessions d'actions.
Le workflow inclut :
1. **Exploration des données** : analyse des sessions, des navigateurs utilisés et des comportements utilisateur.
2. **Construction des features** : extraction de statistiques classiques et TF-IDF sur les actions des sessions.
3. **Analyse des corrélations** : étude de la relation entre les features et la variable cible.
4. **Sélection automatique du meilleur modèle de classification** : tests de plusieurs modèles et évaluation des performances.
5. **Analyse du modèle sélectionné** : importance des variables, matrices de confusion et visualisations.
6. **Réentraînement complet et génération de la soumission** : prédiction sur le jeu de test.

L’ensemble du code est écrit en **Python**, avec des visualisations via **Matplotlib** et **Seaborn**, et des modèles ML via **Scikit-learn**.

---

## Contenu du dépôt

| Fichier/Dossier               | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `BE1_data_science_ECL_2025.py` | Script principal pour le traitement, l’analyse, et la modélisation.       |
| `requirements.txt`            | Liste des packages Python nécessaires.                                      |
| `submission.csv`              | Fichier généré avec les prédictions finales.                                |
| `README.md`                   | Ce fichier.                                                                 |
| `test.csv`                    | Le jeu de test. Le jeu d'entraînement (trop volumineux) n'est pas fourni.   |

---

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/<votre-nom-utilisateur>/<nom-du-repo>.git
   cd <nom-du-repo>
   ```

2. **Créer un environnement Python** :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux / macOS
   venv\Scripts\activate     # Windows
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

---

## Utilisation

1. Placer vos fichiers de données `train.csv` et `test.csv` dans le dossier racine (ou modifier le chemin dans le script).
2. Lancer le script principal :
   ```bash
   python BE1_data_science_ECL_2025.py
   ```

Le script produit :
- Une analyse exploratoire et des visualisations.
- Les features extraites et normalisées.
- Les modèles entraînés et évalués.
- Le meilleur modèle sélectionné et réentraîné.
- Le fichier `submission.csv` pour les prédictions finales.

---

## Structure du code

| Partie                     | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| **PARTIE I**               | Exploration des données et visualisation initiale.                         |
| **PARTIE II**              | Construction des features (classiques + TF-IDF).                           |
| **PARTIE III**             | Statistiques simples et visualisation des features.                        |
| **PARTIE III BIS**         | Corrélation entre features et variable cible.                              |
| **PARTIE IV**              | Sélection automatique du meilleur modèle de classification.                |
| **PARTIE V**               | Analyse détaillée du meilleur modèle (matrices de confusion, importance des variables). |
| **PARTIE VI**              | Réentraînement complet et génération de la soumission.                     |

Chaque section est commentée pour que le fonctionnement soit compréhensible rapidement.

---

## Auteurs

- Groupe **Théo forever** : Julien Durand, Laurène Cristol, Théo Florence
