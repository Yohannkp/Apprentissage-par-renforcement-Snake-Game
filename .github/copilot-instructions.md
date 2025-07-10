# Copilot Instructions pour le Projet Snake RL

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Contexte du projet
Ce projet implémente un agent d'apprentissage par renforcement (Deep Q-Learning) qui apprend à jouer au jeu Snake.

## Technologies utilisées
- **Python** : Langage principal
- **PyTorch** : Framework de deep learning pour l'agent DQN
- **Pygame** : Création de l'environnement Snake
- **NumPy** : Calculs numériques
- **Matplotlib** : Visualisation des résultats d'entraînement

## Structure du projet
- `env/` : Environnement Snake avec pygame
- `agent/` : Agent DQN avec réseau de neurones et replay buffer
- `train.py` : Script d'entraînement
- `test.py` : Script de test et visualisation
- `utils.py` : Fonctions utilitaires

## Directives pour Copilot
1. **Respecter l'architecture RL** : Toujours implémenter les méthodes standard (reset, step, render)
2. **Optimiser les performances** : Utiliser des tensors PyTorch efficacement
3. **Documenter les hyperparamètres** : Commenter les choix d'architecture et paramètres
4. **Gestion des états** : Représenter l'état du jeu de manière optimale pour l'apprentissage
5. **Reward design** : Concevoir des récompenses qui encouragent un bon gameplay
