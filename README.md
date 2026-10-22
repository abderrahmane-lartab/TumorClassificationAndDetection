# Analyse de tumeurs cérébrales par Deep Learning

> **Statut du document :** feuille de route du projet. Les étapes ci-dessous sont des objectifs de travail, et non la déclaration que tous les composants sont déjà implémentés.

## 1. Présentation

Ce projet vise à construire un prototype d’aide à l’analyse d’images cérébrales par IRM, avec deux approches complémentaires :

- **Classification multi-classes** avec TensorFlow/Keras et un modèle pré-entraîné compatible avec GoogLeNet (Inception v1), adapté à quatre classes : `glioma`, `meningioma`, `notumor` et `pituitary`.
- **Détection et localisation** des tumeurs avec YOLOv8, à partir d’images et d’annotations sous forme de boîtes englobantes.

Le pipeline doit également permettre de tracer les expériences avec MLflow, d’orchestrer les étapes avec Apache Airflow, d’exposer les prédictions par FastAPI et de les présenter dans une interface Streamlit.

**Usage prévu :** démonstrateur pédagogique et outil d’assistance à l’évaluation. Ce projet ne constitue pas un dispositif médical et ses prédictions ne remplacent ni l’avis d’un spécialiste ni un diagnostic clinique.

## 2. Objectifs et critères de réussite

- Préparer et contrôler séparément les données de classification et celles de détection.
- Éviter les fuites de données entre entraînement, validation et test.
- Entraîner, comparer et évaluer les deux modèles avec des métriques adaptées à leur tâche.
- Enregistrer les paramètres, métriques et artefacts utiles de chaque expérience dans MLflow.
- Automatiser les étapes reproductibles du pipeline à l’aide d’un DAG Airflow.
- Rendre les deux modèles accessibles via des endpoints FastAPI et les utiliser depuis Streamlit.
- Documenter les instructions d’installation, d’exécution, d’évaluation et les limites du prototype.

Le projet sera considéré comme prêt pour la démonstration lorsque les deux parcours de prédiction fonctionneront de bout en bout, que les résultats de test seront documentés, que les expériences seront consultables et que les erreurs d’entrée seront gérées explicitement.

## 3. Planification sur 10 jours

Période prévue : **du 19 au 30 octobre 2026**. Le calendrier ci-dessous répartit le travail sur dix jours ouvrés ; il laisse le week-end des 24 et 25 octobre hors du décompte.

| Jour | Chantier | Étapes et résultat attendu |
|---|---|---|
| **J1 — 19/10** | Cadrage et environnement | Confirmer les sources de données, les chemins, les licences et les ressources disponibles. Vérifier l’environnement Python et les dépendances. Définir les conventions de configuration, les graines aléatoires et les artefacts à conserver. |
| **J2 — 20/10** | Audit des données | Examiner les dossiers et les classes. Contrôler les extensions, fichiers illisibles, doublons potentiels et distributions par classe. Confirmer le format et l’emplacement des annotations YOLO. Produire un bilan des données avant entraînement. |
| **J3 — 21/10** | Préparation classification | Construire des ensembles stratifiés train/validation/test en proportion **70/15/15**. Vérifier les effectifs par classe et par ensemble. Préparer le redimensionnement, le prétraitement du modèle et les transformations d’augmentation réservées au train. |
| **J4 — 22/10** | Modèle de classification | Sélectionner une implémentation de GoogLeNet/Inception v1 compatible avec TensorFlow/Keras et vérifier la disponibilité de poids pré-entraînés appropriés. Adapter la tête à quatre classes, définir loss/optimiseur/taux d’apprentissage, puis lancer un premier entraînement de référence. |
| **J5 — 23/10** | Évaluation classification | Ajuster un nombre limité de configurations. Évaluer sur validation puis une seule fois sur test avec accuracy, précision, rappel, F1-score, rapport par classe et matrice de confusion. Examiner les erreurs, enregistrer le modèle et les résultats retenus. |
| **J6 — 26/10** | Préparation détection | Construire les ensembles `train`, `valid` et `test`. Associer chaque image à son label par nom de base, exclure et comptabiliser les paires incomplètes, valider le format YOLO et vérifier les classes ainsi que les coordonnées des boîtes. Générer `data.yaml`. |
| **J7 — 27/10** | Modèle YOLOv8 | Visualiser des images avec annotations, puis lancer un entraînement de référence avec une taille d’image et un nombre d’époques adaptés aux ressources disponibles. Documenter les paramètres d’augmentation et conserver le checkpoint et les journaux. |
| **J8 — 28/10** | Évaluation détection et MLflow | Évaluer le meilleur modèle sur validation puis test avec précision, rappel, mAP@0.5 et mAP@0.5:0.95. Comparer les expériences. Enregistrer paramètres, métriques, versions et artefacts des deux tâches dans MLflow. |
| **J9 — 29/10** | API, interface et orchestration | Implémenter et tester les endpoints FastAPI de classification et de détection, puis les appeler depuis Streamlit. Finaliser le DAG Airflow : préparation/validation, entraînements, évaluations et sauvegarde. Tester les erreurs et les statuts de tâches. |
| **J10 — 30/10** | Validation et soutenance | Exécuter le parcours complet sur un petit jeu de données puis sur les modèles retenus. Vérifier la reproductibilité, les résultats, la documentation et la présentation. Préparer une démonstration de 10 minutes et une revue technique de 10 minutes ; réserver 10 minutes à une modification ou mise en situation. |

### Jalons de contrôle

1. **Après J2 :** la structure réelle des données et les annotations sont comprises et leurs problèmes sont consignés.
2. **Après J5 :** le modèle de classification est sauvegardé et son évaluation indépendante est disponible.
3. **Après J8 :** le modèle de détection est sauvegardé et les deux expérimentations sont traçables dans MLflow.
4. **Après J9 :** l’interface communique avec les deux endpoints ; les étapes principales sont représentées dans Airflow.
5. **Après J10 :** démonstration, code, mesures et procédure d’installation sont prêts.

## 4. Démarche détaillée par composant

### 4.1 Données et préparation

#### Classification

1. Charger les images des dossiers de classes et établir les classes à partir des noms de répertoire.
2. Accepter uniquement `.jpeg`, `.jpg`, `.bmp` et `.png`. Signaler les fichiers ignorés ou illisibles plutôt que de les compter comme des images valides.
3. Examiner les distributions de classes avec un graphique de comptage et afficher des exemples pour vérifier visuellement les données.
4. Constituer les ensembles **70 % train, 15 % validation, 15 % test**, de façon stratifiée lorsque les effectifs le permettent. Fixer et documenter la graine de séparation.
5. Vérifier les effectifs par classe dans chaque ensemble et préserver l’indépendance entre les ensembles. Toute augmentation doit intervenir **après** la séparation et uniquement pendant l’entraînement.
6. Redimensionner les images à la taille retenue (par exemple 224 × 224 si elle convient à l’implémentation sélectionnée) et appliquer le prétraitement attendu par le modèle.
7. Appliquer le retournement, le bruit ou le flou uniquement sur train, avec des transformations plausibles pour l’imagerie. En cas de déséquilibre, mesurer l’effet des augmentations et envisager aussi des poids de classes ; les transformations seules ne garantissent pas un équilibrage.
8. Utiliser le chargement par lots, mélanger train uniquement et garder validation/test sans augmentation aléatoire.

L’implémentation exécutable de cette préparation est dans [notebooks/classification_data.py](./notebooks/classification_data.py). La graine de séparation est fixée à `42`. Depuis la racine du dépôt, lancer `python notebooks/classification_data.py`.

#### Détection YOLOv8

1. Repérer les images et annotations, puis les associer à partir du nom de base, sans extension.
2. Répartir les paires image/label entre `train`, `valid` et `test` selon la répartition retenue et documenter cette répartition.
3. Exclure et comptabiliser les images sans annotation et les annotations sans image correspondante.
4. Contrôler chaque ligne d’annotation : identifiant de classe entier et connu, coordonnées `x_center y_center width height` normalisées dans `[0, 1]`, largeur/hauteur positives et nombre de champs attendu.
5. Vérifier visuellement un échantillon représentatif des boîtes et classes après conversion éventuelle.
6. Générer `data.yaml` avec les chemins des ensembles, le nombre de classes et leurs noms. Compter les images et labels de chaque ensemble.
7. Garder test isolé du choix des hyperparamètres. Ne l’utiliser que pour l’évaluation finale.

### 4.2 Classification TensorFlow/Keras

1. Valider l’implémentation de GoogLeNet/Inception v1 retenue : compatibilité TensorFlow/Keras, entrée attendue, prétraitement associé et origine des poids.
2. Remplacer la tête de classification par une sortie à quatre classes et choisir une fonction de perte cohérente avec le format des labels.
3. Établir une référence reproductible, puis comparer quelques configurations d’hyperparamètres (taux d’apprentissage, optimiseur, taille de batch, époques et stratégie de gel/dégel des couches).
4. Suivre les métriques train/validation et utiliser des callbacks adaptés, par exemple arrêt anticipé et sauvegarde du meilleur modèle.
5. Mesurer sur le jeu de test accuracy, précision, rappel et F1-score, de préférence en complément par classe et en macro-moyenne. Fournir une matrice de confusion et analyser les confusions.
6. Sauvegarder le modèle sélectionné, le mapping classe-index et les informations de prétraitement nécessaires à l’inférence.

### 4.3 Détection YOLOv8

1. Confirmer que le dataset est bien au format attendu par Ultralytics et que les identifiants de classes concordent avec `data.yaml`.
2. Lancer un entraînement de référence, puis tester des ajustements limités et justifiés selon le temps et le matériel disponibles.
3. Documenter les paramètres principaux : poids de départ, taille d’image, batch, époques, optimiseur/taux d’apprentissage si configurés, graine et augmentations.
4. Évaluer précision, rappel, mAP@0.5 et mAP@0.5:0.95 sur validation et test. Vérifier les résultats par classe et inspecter les prédictions visuellement.
5. Conserver les poids retenus et les fichiers de configuration nécessaires pour reproduire l’inférence.

### 4.4 Suivi MLflow

Pour chaque entraînement, associer l’exécution à une expérience et enregistrer :

- paramètres, graine, configuration des données et versions des bibliothèques ;
- métriques train/validation/test clairement distinguées ;
- artefacts utiles : courbes, matrice de confusion, rapport de classification, configuration YOLO et poids du modèle retenu ;
- identifiant et version du modèle afin de permettre sa réutilisation.

Comparer les expériences sur des métriques adaptées à la tâche, sans sélectionner un modèle sur les seules performances du test. Documenter la configuration finale et la raison du choix.

### 4.5 Orchestration Airflow

Créer un DAG lisible avec des tâches séparées et dépendances explicites, dans cet ordre logique :

1. vérification et préparation des données ;
2. validation de qualité et de structure ;
3. entraînement du classifieur ;
4. entraînement du détecteur ;
5. évaluation des modèles et journalisation MLflow ;
6. sauvegarde et publication des artefacts retenus.

Prévoir des journaux exploitables, des statuts d’échec visibles et des paramètres d’exécution reproductibles. Les tâches longues doivent référencer des chemins/artefacts persistants et éviter de transférer les poids volumineux dans les métadonnées du DAG.

### 4.6 API FastAPI

Exposer deux parcours distincts : classification et détection. Pour chaque endpoint, définir dans la documentation OpenAPI :

- méthode, chemin et format du fichier attendu ;
- structure de la réponse (classe/scores ou détections/boîtes/classes/confiances) ;
- validation des fichiers, limites de taille et types MIME acceptés ;
- erreurs explicites pour fichier invalide, modèle manquant ou échec d’inférence.

Charger les modèles une fois au démarrage lorsque possible. Garder le prétraitement de l’API strictement cohérent avec celui utilisé pendant l’entraînement.

### 4.7 Interface Streamlit

Créer une interface permettant de téléverser une image et de choisir entre :

- **Classification :** afficher la classe prédite et les scores par classe ;
- **Détection :** afficher l’image annotée, les boîtes, les classes et les scores de confiance.

L’interface appelle FastAPI par HTTP, indique clairement les erreurs de connexion et ne présente pas une réponse échouée comme une prédiction réussie.

## 5. Organisation cible du dépôt

La structure existante comprend déjà des répertoires pour `src`, `models`, `api`, `ui`, `airflow`, `data`, `notebooks` et `tests`. Les responsabilités prévues sont :

```text
.
├── airflow/       # DAG et orchestration du pipeline
├── api/           # application et endpoints FastAPI
├── data/          # données locales (non versionnées) et/ou instructions de préparation
├── models/        # artefacts de modèles générés
├── notebooks/     # exploration et prototypage ; code stabilisé à déplacer dans src
├── src/
│   ├── classification/  # préparation, entraînement et évaluation classification
│   └── detection/       # préparation, entraînement et évaluation YOLO
├── tests/
│   ├── classification/ # tests du pipeline de classification
│   └── detection/       # tests du format et des données de détection
├── ui/            # interface Streamlit
├── requirements.txt
└── README.md
```

Cette structure est une cible d’organisation, pas une affirmation que chaque composant est déjà implémenté. Les données et les modèles entraînés doivent rester distincts du code source ; vérifier les règles de versionnement et les licences avant de publier des données ou poids.

## 6. Validation et critères d’acceptation

- [ ] Les extensions non autorisées et images illisibles sont écartées avec un bilan explicite.
- [ ] Les séparations sont reproductibles, sans augmentation avant séparation ni fuite entre train/validation/test.
- [ ] Les augmentations de classification ne s’appliquent qu’à train.
- [ ] Les annotations YOLO et leurs paires image/label sont validées et comptabilisées.
- [ ] Les métriques de test des deux modèles sont produites et interprétées, y compris par classe.
- [ ] Les modèles, mappings de classes, configurations et résultats nécessaires à l’inférence sont sauvegardés.
- [ ] Chaque expérience retenue peut être retrouvée dans MLflow avec ses paramètres et métriques.
- [ ] Le DAG s’exécute dans l’ordre prévu et rend les erreurs visibles.
- [ ] FastAPI renvoie les formats documentés et gère les entrées incorrectes.
- [ ] Streamlit affiche les résultats des deux endpoints et leurs erreurs éventuelles.
- [ ] Le README final inclut les prérequis et commandes d’installation, de lancement et de démonstration vérifiés sur l’environnement choisi.

## 7. Risques et décisions à surveiller

| Risque | Mesure prévue |
|---|---|
| Implémentation ou poids GoogLeNet incompatibles avec TensorFlow/Keras | Vérifier tôt l’API, les dimensions, le prétraitement et la licence ; documenter la source des poids. |
| Données manquantes, corrompues ou déséquilibrées | Établir un rapport qualité avant entraînement et ne pas masquer les exclusions. |
| Fuite de données ou augmentation appliquée au mauvais ensemble | Séparer d’abord, puis augmenter exclusivement train ; garder test à l’écart de la sélection. |
| Annotations YOLO mal formées ou classes divergentes | Valider automatiquement les valeurs et inspecter les boîtes sur images avant d’entraîner. |
| Temps ou mémoire insuffisants | Faire une exécution de référence courte, choisir batch/taille d’image adaptés et enregistrer les ressources/paramètres. |
| Résultats élevés mais mauvaise généralisation | Comparer train/validation/test, consulter les métriques par classe et contrôler les erreurs visuellement. |
| Dépendance trop lourde à Airflow pour un poste local | Vérifier les ressources et la procédure de lancement tôt ; garder les scripts d’entraînement exécutables indépendamment du DAG. |
| Usage clinique interprété à tort | Présenter le projet comme un prototype non clinique et afficher ses limites dans l’interface et la documentation. |

## 8. Livrables pour la soutenance

- Code source organisé et publié sur GitHub.
- Scripts ou modules de préparation et d’entraînement pour les deux modèles.
- Résultats d’évaluation sur test, matrices/rapports et comparaison des expériences.
- Expériences MLflow consultables et artefacts des modèles retenus.
- DAG Airflow et preuve de son exécution.
- API FastAPI documentée avec ses deux fonctionnalités.
- Interface Streamlit connectée à l’API.
- Instructions reproductibles d’installation et de lancement.
- Démonstration de 10 minutes, revue de code/projet de 10 minutes et mise en situation de 10 minutes.

## 9. Source de données

Dataset de détection indiqué dans le cahier des charges : [MRI for Brain Tumor with Bounding Boxes — Kaggle](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes).

Avant utilisation ou publication, vérifier la structure réelle, la correspondance des classes, les modalités d’annotation et les conditions de licence du dataset. Les données ne sont pas supposées être incluses dans ce dépôt.
