# Snake RL - Apprentissage par Renforcement

🐍 **Une IA qui apprend à jouer à Snake via Deep Q-Learning (DQN)**  
⚡ **Optimisé pour GPU NVIDIA (RTX 4060 et plus)**

## 🎯 Objectif

Ce projet implémente un agent d'apprentissage par renforcement qui apprend autonomement à jouer au jeu Snake en utilisant l'algorithme Deep Q-Network (DQN) avec PyTorch. L'entraînement est **accéléré par GPU** pour des performances optimales.

## 🚀 Fonctionnalités

- ✅ **Environnement Snake** personnalisé compatible Gymnasium
- ✅ **Agent DQN** avec target network et experience replay
- ✅ **Accélération GPU** - Détection automatique RTX 4060/RTX 30-40 Series
- ✅ **Entraînement ultra-rapide** - 15-20 min vs 2-3h (10x plus rapide)
- ✅ **Mode de jeu manuel** - Jouez vous-même pour comparer
- ✅ **Tests et comparaisons** de performances détaillés
- ✅ **Analyse** complète avec visualisations
- ✅ **Sauvegarde/Chargement** de modèles optimisés
- ✅ **Notebook pédagogique** complet pour débutants

## 📦 Technologies

- **Python 3.8+**
- **PyTorch** - Deep learning framework (CUDA optimisé)
- **Pygame** - Environnement de jeu
- **NumPy** - Calculs numériques  
- **Matplotlib** - Visualisation
- **Gymnasium** - API d'environnement RL
- **CUDA** - Accélération GPU NVIDIA

## 🎮 GPU Support

### GPU Supportés
- ✅ **RTX 40 Series** (4060, 4070, 4080, 4090) - Recommandé
- ✅ **RTX 30 Series** (3060, 3070, 3080, 3090)
- ✅ **RTX 20 Series** (2060, 2070, 2080)
- ✅ **GTX 16 Series** (1660, 1650) - Performance réduite
- ⚠️ **CPU seulement** - Fonctionnel mais plus lent

### Performances GPU vs CPU
| Configuration | Entraînement 100 épisodes | Entraînement 2000 épisodes |
|--------------|---------------------------|----------------------------|
| **RTX 4060** | **30-60 sec** | **15-20 min** |
| **RTX 3070** | **25-50 sec** | **12-18 min** |
| **CPU seul** | 5-10 min | 2-3 heures |

## 🏗️ Structure du Projet

```
snake-rl/
├── env/                    # Environnement Snake
│   └── snake_game.py      # Classe SnakeGame (Gymnasium compatible)
├── agent/                  # Agent d'apprentissage
│   └── dqn_agent.py       # Agent DQN avec GPU optimization
├── models/                 # Modèles sauvegardés (créé automatiquement)
├── .vscode/                # Configuration VS Code
│   ├── tasks.json         # Tâches d'entraînement/test
│   └── settings.json      # Paramètres optimisés
├── train.py               # Script d'entraînement (GPU optimisé)
├── quick_train.py         # Entraînement rapide (100 épisodes)
├── test.py                # Script de test et évaluation
├── test_gpu.py            # Test des performances GPU
├── play_manual.py         # Mode de jeu humain
├── demo.py                # Démonstration rapide
├── utils.py               # Fonctions utilitaires et analyse
├── Snake_RL_Guide_Complet.ipynb  # Notebook pédagogique
└── requirements.txt       # Dépendances Python avec CUDA
```

## ⚡ Installation Rapide

### Prérequis
- **Python 3.8+**
- **GPU NVIDIA** avec drivers récents (recommandé)
- **8 GB RAM** minimum (16 GB recommandé)

### Installation Automatique

1. **Cloner le projet**
```bash
git clone <votre-repo>
cd snake-rl
```

2. **Installer les dépendances (avec support GPU)**
```bash
pip install -r requirements.txt
```
*Note : Installe automatiquement PyTorch avec CUDA 11.8*

3. **Vérifier la configuration GPU**
```bash
python test_gpu.py
```

4. **Test rapide (2 minutes)**
```bash
python quick_train.py
```

5. **Entraînement complet (15-20 min avec GPU)**
```bash
python train.py
```

### VS Code (Recommandé)

Le projet inclut des tâches VS Code prêtes à l'emploi :

1. **Ouvrir dans VS Code**
2. **Ctrl+Shift+P** → "Tasks: Run Task"
3. **Choisir :**
   - `Demo - Test Installation`
   - `Quick Train - 100 Episodes`  
   - `Full Training - 2000 Episodes`
   - `Test GPU RTX 4060`
   - `Test Agent`

## 🎮 Utilisation

### Entraînement

```bash
# Entraînement standard (2000 épisodes)
python train.py

# Les modèles sont sauvegardés dans models/
# - snake_dqn.pth : modèle final
# - snake_dqn_best.pth : meilleur modèle
```

### Test et Évaluation

```bash
python test.py
```

Menu interactif avec options :
1. **Test visuel** - Voir l'agent jouer
2. **Comparaison** - Agent entraîné vs aléatoire  
3. **Mode interactif** - Contrôle manuel du test
4. **Analyse** - Graphiques d'entraînement détaillés

### Exemples d'Utilisation

```python
# Charger et utiliser un agent entraîné
from env.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent

env = SnakeGame()
agent = DQNAgent()
agent.load_model("models/snake_dqn_best.pth")

state, _ = env.reset()
action = agent.get_action(state, training=False)
```

## 🧠 Architecture de l'Agent

### Réseau de Neurones (DQN)
- **Entrée** : État du jeu (11 features)
  - Dangers (straight, left, right)
  - Direction actuelle (4 directions)
  - Position relative de la nourriture (4 directions)
- **Couches cachées** : 256 → 256 neurones (ReLU)
- **Sortie** : Q-values pour 3 actions (tout droit, droite, gauche)

### Algorithme DQN
- ✅ **ε-greedy** exploration policy
- ✅ **Target network** pour la stabilité
- ✅ **Experience replay** buffer (10k expériences)
- ✅ **Gradient clipping** pour éviter l'instabilité

### Hyperparamètres par Défaut
```python
learning_rate = 0.001
gamma = 0.9              # Facteur de discount
epsilon = 1.0 → 0.01     # Exploration décroissante
batch_size = 32
memory_size = 10000
target_update = 1000     # Fréquence de mise à jour du target network
```

## 📊 Résultats Attendus

Un agent bien entraîné devrait atteindre :
- **Score moyen** : 10-20+ (vs ~1 pour un agent aléatoire)
- **Score maximum** : 30-50+
- **Convergence** : visible après 500-1000 épisodes

## 🎛️ Configuration Avancée

### Modification des Hyperparamètres

Éditez directement dans `train.py` :

```python
agent = DQNAgent(
    lr=0.0005,           # Learning rate plus faible
    gamma=0.95,          # Facteur de discount plus élevé
    epsilon_decay=0.99,  # Décroissance plus lente
    batch_size=64        # Batch plus grand
)
```

### Reward Shaping

Modifiez la fonction `step()` dans `env/snake_game.py` :

```python
# Ajouter des récompenses intermédiaires
if self.head == self.food:
    reward = 10
else:
    # Récompense pour se rapprocher de la nourriture
    reward = self._get_food_reward()
```

### Architecture du Réseau

Modifiez la classe `DQN` dans `agent/dqn_agent.py` :

```python
class DQN(nn.Module):
    def __init__(self, input_size=11, hidden_size=512, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)  # Couche supplémentaire
        self.linear4 = nn.Linear(hidden_size, output_size)
```

## 🔧 Troubleshooting

### Problèmes Courants

**L'agent n'apprend pas :**
- Vérifiez le learning rate (essayez 0.0001-0.01)
- Augmentez la taille du replay buffer
- Réduisez epsilon_decay pour plus d'exploration

**Entraînement instable :**
- Activez le gradient clipping
- Réduisez le learning rate  
- Augmentez la fréquence de mise à jour du target network

**Performances faibles :**
- Vérifiez la représentation de l'état
- Ajustez les récompenses (reward shaping)
- Augmentez la taille du réseau

## 🚀 Améliorations Possibles

### Algorithmes RL Avancés
- [ ] **Double DQN** - Réduction du biais d'optimisation
- [ ] **Dueling DQN** - Séparation Value/Advantage
- [ ] **Prioritized Experience Replay** - Échantillonnage intelligent
- [ ] **Rainbow DQN** - Combinaison de toutes les améliorations

### Environnement
- [ ] **Obstacles** - Complexifier le jeu
- [ ] **Multi-snake** - Environnement multi-agents
- [ ] **Variantes** - Différentes tailles, vitesses
- [ ] **Curriculum Learning** - Difficulté progressive

### Interface
- [ ] **Dashboard Streamlit** - Interface web
- [ ] **Enregistrement vidéo** - Sauvegarder les parties
- [ ] **API REST** - Serveur pour démo
- [ ] **Comparaison en ligne** - Leaderboard

## 📈 Métriques et Analyse

Le projet inclut des outils d'analyse avancés :

```python
from utils import calculate_learning_metrics, plot_comparison

# Analyser les résultats
scores = np.load("models/training_scores.npy")
metrics = calculate_learning_metrics(scores)

# Comparer plusieurs expériences
experiments = {
    'DQN_base': scores1,
    'DQN_tuned': scores2
}
plot_comparison(experiments)
```

## 🤝 Contribution

Les contributions sont bienvenues ! 

1. Fork le projet
2. Créez une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add some AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir `LICENSE` pour plus de détails.

## 🔗 Références

- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch RL Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

---

**Créé avec ❤️ pour l'apprentissage par renforcement**
