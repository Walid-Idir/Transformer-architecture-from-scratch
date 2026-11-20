# 🚀 Transformer Architecture from Scratch (PyTorch)
### *De la modélisation séquentielle (RNN, LSTM) aux architectures Transformers (GPT, BERT)*

**Auteur :** Walid Idir  
**Technologies :** Python, PyTorch, Hugging Face Transformers, Pandas, NumPy, Matplotlib

---

## 📌 Présentation du Projet
Ce projet propose une étude comparative approfondie et une implémentation complète des réseaux de neurones pour le traitement des séquences et des séries temporelles, ainsi que l'implémentation *from scratch* des deux grands paradigmes de l'architecture **Transformer** : **GPT** (décodeur autorégressif) et **BERT** (encodeur bidirectionnel).

---

## 📂 Contenu du Repository

```text
├── TP_Architecture_Transformer_Complet.ipynb  # Notebook complet avec sorties, métriques et graphiques
├── architecture_transformer (1).ipynb         # Notebook mis à jour avec les exercices résolus
├── corpus.txt                                 # Corpus textuel d'entraînement pour Mini-GPT
├── .gitignore                                 # Exclusion des datasets volumineux (>100 Mo)
└── README.md                                  # Documentation du projet
```

> **Note sur le dataset d'énergie :** Le fichier `household_power_consumption.txt` (~133 Mo) issu du dépôt UCI Machine Learning Repository n'est pas versionné directement sur GitHub en raison de sa taille (> 100 Mo). Le notebook le charge automatiquement depuis le répertoire local.

---

## 🔬 Première Partie : Fondements et Modèles de Base

Sur une série temporelle synthétique $y = \sin(x)$ découpée en séquences de 10 pas de temps pour prédire la valeur suivante ($t+10$) :

1. **RNN Simple (`SimpleRNN`)** :
   - État caché récurrent de dimension 16 mis à jour pas à pas.
   - **Erreur MSE test :** `0.000114`
2. **LSTM (`SimpleLSTM`)** :
   - Cellule mémoire régulée par 3 portes (oubli, entrée, sortie) pour surmonter la disparition du gradient.
   - **Erreur MSE test :** `0.000069`
3. **Transformer (`MiniTransformer`)** :
   - Projection linéaire (1 $\to$ 16), encodage positionnel appris, auto-attention multi-têtes (2 têtes) sans récurrence temporelle.
   - **Erreur MSE test :** `0.000048` *(meilleure performance parmi les trois modèles)*
4. **Modèle de type GPT (GPT-2)** :
   - Génération de texte autorégressive avec `gpt2` (Hugging Face) via échantillonnage probabiliste (`temperature=0.8`, `top_p=0.9`).
5. **Modèle de type BERT (BERT multilingue)** :
   - Résolution de masque de langage (*Masked Language Modeling*) avec `bert-base-multilingual-cased` pour retrouver un mot masqué à partir du contexte gauche et droite.

---

## 💡 Deuxième Partie : Exercices Pratiques d'Application

### ⚡ Exercice 1 : Prédiction de la consommation électrique horaire (`EnergyTransformer`)
- **Données :** *Individual Household Electric Power Consumption* (UCI) agrégé à l'échelle horaire (34 151 heures exploitables).
- **Méthodologie :** Découpage chronologique 70% train / 15% val / 15% test, normalisation $z$-score basée uniquement sur le train (évitement strict de *data leakage*).
- **Architecture :** Projection 1 $\to$ 32, embedding positionnel des 24 heures, 2 couches d'encodeur Transformer à 4 têtes ($d_{\text{model}}=32$, $\text{dim}_{\text{ff}}=128$), régression linéaire 32 $\to$ 1.
- **Résultats comparatifs (sur les données de test) :**

| Modèle / Référence | MAE (kW) | RMSE (kW) | Gain par rapport aux baselines |
| :--- | :---: | :---: | :--- |
| **EnergyTransformer** | **0.2974** | **0.4789** | **Meilleure performance globale** |
| Persistance ($t-1$) | 0.3218 | 0.5312 | +10.9% d'erreur RMSE |
| Référence journalière ($t-24$) | 0.4485 | 0.7250 | +51.4% d'erreur RMSE |

- Visualisation des observations et prédictions sur 7 jours consécutifs (168 heures).

---

### 🔤 Exercice 2 : Mini-GPT *from scratch* (Niveau caractère)
- **Objectif :** Implémenter un Transformer autorégressif de A à Z.
- **Architecture :**
  - Normalisation Pre-LN (`LayerNorm`) avant l'attention.
  - Auto-attention multi-têtes avec **masque causal triangulaire supérieur strict** (`torch.triu(..., diagonal=1)`).
  - Connexion résiduelle.
  - Seconde normalisation Pre-LN et MLP ($64 \to 256 \to 64$ avec activation `GELU`).
  - Seconde connexion résiduelle.
- **Vérification de non-triche causale :** Différence maximale sur les positions antérieures égale à **$0.00000000$** lors de la modification d'un token futur.
- **Génération :** Échantillonnage de 200 caractères sous 3 régimes de température :
  - $\tau = 0.5$ : Très conservateur et structuré.
  - $\tau = 1.0$ : Équilibre créativité/cohérence.
  - $\tau = 1.5$ : Forte entropie et diversité lexicale.

---

### 🎭 Exercice 3 : Mini-BERT *from scratch* (Masked Language Modeling)
- **Objectif :** Implémenter un Transformer bidirectionnel pour le remplissage de masques.
- **Gestion des tokens spéciaux :** `PAD=0, UNK=1, CLS=2, SEP=3, MASK=4`.
- **Stratégie de masquage (conforme au papier BERT) :** 15 % des tokens ordinaires sélectionnés (80 % `[MASK]`, 10 % mot aléatoire, 10 % inchangé).
- **Perte ciblée :** Calculée uniquement sur les positions sélectionnées avec `CrossEntropyLoss(ignore_index=-100)`.
- **Vérification d'invariance au padding :** Différence maximale égale à **$0.00000000$** lors de l'altération des tokens de padding sous masque.
- **Démonstration :** Prédiction réussie du mot masqué sur de nouvelles phrases (*"le chat dort sur le [MASK] du salon"* $\to$ `canape` avec 86.42% de probabilité).

---

## ⚙️ Installation et Exécution

1. **Cloner le repository :**
   ```bash
   git clone https://github.com/Walid-Idir/Transformer-architecture-from-scratch.git
   cd Transformer-architecture-from-scratch
   ```

2. **Installer les dépendances :**
   ```bash
   pip install torch transformers pandas numpy matplotlib scikit-learn
   ```

3. **Lancer le notebook :**
   ```bash
   jupyter notebook TP_Architecture_Transformer_Complet.ipynb
   ```
