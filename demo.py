#!/usr/bin/env python3
"""
Script de démonstration pour vérifier l'installation et tester l'environnement Snake.
"""

import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import pygame
    import numpy as np
    import matplotlib.pyplot as plt
    import gymnasium as gym
    from env.snake_game import SnakeGame
    from agent.dqn_agent import DQNAgent
    
    print("=== VÉRIFICATION DE L'INSTALLATION ===")
    print(f"✅ Python: {sys.version}")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ PyGame: {pygame.version.ver}")
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ Gymnasium: {gym.__version__}")
    
    # Test de l'environnement
    print("\n=== TEST DE L'ENVIRONNEMENT ===")
    env = SnakeGame(width=400, height=300, block_size=20, speed=5)
    print(f"✅ Environnement Snake créé")
    print(f"   - Espace d'action: {env.action_space}")
    print(f"   - Espace d'observation: {env.observation_space}")
    
    # Test de reset
    state, info = env.reset()
    print(f"✅ Reset réussi, état: {state.shape}")
    
    # Test de quelques steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()  # Action aléatoire
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"✅ Steps de test réussis, reward total: {total_reward}")
    env.close()
    
    # Test de l'agent
    print("\n=== TEST DE L'AGENT ===")
    agent = DQNAgent(state_size=11, action_size=3)
    print(f"✅ Agent DQN créé")
    print(f"   - Device: {agent.device}")
    print(f"   - Paramètres du modèle: {agent.get_model_info()}")
    
    # Test d'une action
    test_state = np.random.random(11)
    action = agent.get_action(test_state, training=False)
    print(f"✅ Action test: {action}")
    
    print("\n🎉 INSTALLATION RÉUSSIE!")
    print("\nPour commencer l'entraînement, lancez:")
    print("   python train.py")
    print("\nPour tester un agent entraîné, lancez:")
    print("   python test.py")

except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    print("\nVeuillez installer les dépendances:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

except Exception as e:
    print(f"❌ Erreur inattendue: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
