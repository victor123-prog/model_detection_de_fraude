# model_detection_de_fraude
Detection_FRaude_Bancaire
# Détection de fraude bancaire par réseau de neurones

Classification binaire de transactions bancaires (**fraude / légitime**) avec un réseau de neurones **PyTorch**, sur environ 700 000 transactions d'entraînement. Le projet couvre tout le pipeline : prétraitement, expérimentations, choix du modèle et prédictions.

## Résultats

Les fraudes représentent environ **1,7 %** des transactions : l'accuracy seule est trompeuse, on suit donc aussi la précision, le rappel, le F1 et l'AUC (jeu de validation, seuil 0,5).

| Expérience | Entrées | Accuracy | Précision | Rappel | F1 | AUC |
|---|---|---|---|---|---|---|
| 1. Baseline | 8 variables numériques | 0,9836 | 0,6882 | 0,0275 | 0,0528 | n/d |
| 2. Réseau complet | 42 variables | 0,9865 | 0,7592 | 0,2747 | 0,4034 | n/d |
| **Modèle retenu** | **42 variables** | **0,9865** | **0,7543** | **0,2780** | **0,4063** | **0,9355** |

**À retenir** : encoder les variables catégorielles multiplie le rappel par dix, et l'AUC de 0,94 montre que le modèle classe bien les transactions. Le rappel reste modeste au seuil de 0,5 : le déséquilibre des classes n'est pas encore traité (voir *Pistes d'amélioration*).

## Pipeline

1. **Prétraitement** : encodage one-hot (`merchant_category`, `device_type`), encodage entier (`card_type`, `auth_method`, `channel`), conversion des booléens, puis normalisation Min-Max. Le test est transformé avec le scaler appris sur l'entraînement, pour éviter toute fuite d'information.
2. **Modèle** : perceptron multicouche `42 → 60 → 20 → 1` (BatchNorm + ELU, sortie sigmoïde), optimiseur Adam, perte `BCELoss`. Séparation 80 / 20 avec graine fixée.
3. **Expériences** : baseline sur 8 variables, réseau sur 42 variables, réseau plus profond.
4. **Prédictions** : probabilité de fraude pour chaque transaction du test, exportée dans `resultat.csv`.

## Structure du dépôt

```
.
├── fraud_detection_pipeline.ipynb   # notebook complet (prétraitement + modèles + prédictions)
├── README.md
├── train.csv                        # données d'entraînement (à ajouter)
└── test.csv                         # données de test (à ajouter)
```

Fichiers générés à l'exécution : `train_normalized.csv`, `data_test_normalized.csv`, `model_weights.pth`, `resultat.csv` (ainsi que `test_train.csv` et `test_test.csv`, intermédiaires de l'expérience 1).


## Installation et utilisation

```bash
pip install pandas numpy scikit-learn torch torchmetrics tqdm matplotlib jupyter ipywidgets
```

1. Placer `train.csv` et `test.csv` à côté du notebook.
2. Ouvrir `fraud_detection_pipeline.ipynb` avec Jupyter.
3. Exécuter les cellules dans l'ordre. Prévoir plusieurs minutes par entraînement (environ 40 s par époque sur CPU).

## Données

Colonnes principales : montant, délai depuis la dernière transaction, nombre de transactions sur 24 h, distance au domicile, âge de la carte et du client, solde, score de vélocité, score de risque du marchand, canal, méthode d'authentification, type d'appareil, indicateurs (VPN, transaction étrangère, nouveau marchand, etc.). La cible est `is_fraud`.

Source des données : *à compléter (lien et licence)*.

## Limites connues

- Les métriques `torchmetrics` ne sont pas réinitialisées entre deux évaluations : les effectifs des matrices de confusion sont cumulés (les proportions restent valables).
- Dans l'expérience 3 (réseau plus profond), l'optimiseur utilisé était celui du premier modèle : la comparaison de profondeur est à refaire.
- Les prédictions finales sont faites avec `model` (entraînement poursuivi) et non avec `model_choisit`, sans passage préalable en `model.eval()`.

## Pistes d'amélioration

- Pondérer la classe fraude dans la perte (`pos_weight`) ou rééchantillonner.
- Ajuster le seuil de décision avec la courbe précision-rappel.
- Early stopping et suivi de la perte de validation.
- Comparer à des modèles classiques (forêt aléatoire, gradient boosting, SVM) avec `GridSearchCV`.
- Validation croisée, et one-hot pour `card_type`, `auth_method` et `channel`.
