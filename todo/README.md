# Projet fil rouge : votre pipeline MLOps de classification

Ce dossier est un projet de classification **generique et autonome** que vous adaptez a
**votre** cas d'usage. Tout le squelette est ici (package `mlproject`, frontend, Docker,
Airflow, Makefile) ; a vous de le completer au fil des seances.

## Demarche pedagogique

1. **Choisissez une problematique de classification binaire.** Deux classes uniquement
   (ex. fraude/non-fraude, defaut/pas de defaut, spam/non-spam, churn/pas de churn,
   sain/malade...). Vous devez pouvoir expliquer en une phrase qui sont les `1`, qui sont
   les `0`, et pourquoi predire cette cible est utile.
2. **Trouvez vos propres donnees.** Un jeu de donnees tabulaire avec une colonne cible
   binaire, des colonnes numeriques et (optionnellement) categorielles. Placez votre CSV
   dans `data/` (a la racine du projet).
3. **Suivez les etapes proposees dans ce dossier `todo/`.** Chaque fichier contient des
   marqueurs `TODO (Sx-n)` qui correspondent aux etapes d'un enonce de TP (dossier `tp/`).
   Vous completez ces TODO, et seulement ceux-la : le reste du code est fourni et
   fonctionnel.
4. **Vous pouvez aller plus loin** (sections "Pour aller plus loin" des TP, idees
   personnelles), mais **au minimum vous faites tous les exercices du `todo/`**.

> Le seul fichier a modifier pour brancher votre dataset est `mlproject/config.py`
> (voir `tp/TP_S0_projet_personnel.md`). `data.py` et `features.py` lisent automatiquement
> votre configuration : vous n'avez pas a les toucher.

## Mise en route

```bash
make -C todo install      # installe les dependances (uv)
export PYTHONPATH=todo     # rend le package `mlproject` importable
```

Adaptez ensuite `todo/mlproject/config.py` a votre dataset (TP S0), puis verifiez :

```bash
PYTHONPATH=todo python -m mlproject.train   # doit afficher f1=... roc_auc=...
```

## Contenu du dossier

```
todo/
  README.md                ce fichier
  Makefile                 cibles d'installation fournies, les autres a completer
  mlproject/
    config.py              configuration a adapter a votre dataset   <- TP S0
    data.py                chargement + split (fourni, generique)
    features.py            pre-processing ColumnTransformer (fourni, generique)
    evaluation.py          summary plot SHAP (fourni, generique)
    train.py               squelette baseline MLflow                  <- TP S5
    train_optuna.py        squelette optimisation Optuna + Registry    <- TP S6
    train_models.py        squelette GridSearchCV + SHAP               <- TP S7
    api.py                 squelette API FastAPI                       <- TP S12
  frontend/app.py          squelette frontend Streamlit                <- TP S14 bis
  docker/Dockerfile.train  squelette image d'entrainement              <- TP S8
  docker/Dockerfile.api    image de l'API (fournie, support compose)
  docker/Dockerfile.frontend  image du frontend (fournie, support compose)
  docker-compose.yml       squelette d'orchestration                   <- TP S14
  dags/retrain_dag.py      squelette DAG de re-entrainement            <- TP S17
```

## Feuille de route (exercices a realiser)

| Seance | Enonce                          | Fichier a completer            | Objectif                                  |
|--------|---------------------------------|--------------------------------|-------------------------------------------|
| S0     | `tp/TP_S0_projet_personnel.md`  | `mlproject/config.py`          | Brancher votre dataset                    |
| S5     | `tp/TP_S5_mlflow.md`            | `mlproject/train.py`           | Suivi d'experiences MLflow                |
| S6     | `tp/TP_S6_optuna.md`           | `mlproject/train_optuna.py`    | Optimisation Optuna + Model Registry      |
| S7     | `tp/TP_S7_automl_shap.md`      | `mlproject/train_models.py`    | Comparaison de modeles (GridSearchCV) + SHAP |
| S8     | `tp/TP_S8_docker.md`           | `docker/Dockerfile.train`      | Conteneuriser l'entrainement              |
| S12    | `tp/TP_S12_fastapi.md`         | `mlproject/api.py`             | Exposer le modele via une API FastAPI     |
| S14    | `tp/TP_S14_docker_compose.md`  | `docker-compose.yml`           | Orchestrer la stack                       |
| S14bis | `tp/TP_S14_bis_streamlit.md`   | `frontend/app.py`              | Frontend de test                          |
| S17    | `tp/TP_S17_airflow.md`         | `dags/retrain_dag.py`          | Planifier le re-entrainement              |

Toutes les commandes s'executent depuis la racine du projet avec `PYTHONPATH=todo`
(ex. `PYTHONPATH=todo python -m mlproject.train`).

## Suivi sur GitHub (obligatoire)

L'avancement des TP doit etre **documente et pousse sur GitHub tout au long du cours** :

- Creez un depot GitHub pour votre projet (public ou prive).
- A chaque seance, **committez et poussez** votre travail (code complete, notes,
  resultats). Des commits reguliers et des messages clairs font partie de l'evaluation.
- Documentez l'evolution de vos TP : tenez ce `README.md` (ou un journal de bord) a jour
  avec votre problematique, votre jeu de donnees et ce que vous avez realise a chaque
  seance.
- **Ajoutez l'enseignant comme collaborateur du depot** :
  [github.com/lewishkpv](https://github.com/lewishkpv) (compte `lewishkpv`). Cet acces
  doit rester actif pendant toute la duree du cours pour permettre le suivi de vos
  travaux. Sur GitHub : *Settings > Collaborators > Add people > `lewishkpv`*.

> Ne versionnez jamais de secrets ni d'artefacts lourds : le `.gitignore` du projet
> exclut deja `.env`, `.venv/`, `mlruns/`, les modeles `*.joblib`, les caches, etc.
