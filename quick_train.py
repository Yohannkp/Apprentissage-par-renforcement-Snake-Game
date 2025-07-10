#!/usr/bin/env python3
"""
Script d'entraînement rapide pour tester le projet (100 épisodes seulement).
Utilisez train.py pour un entraînement complet.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

def quick_train(n_episodes=100):
    """Entraînement rapide pour démonstration."""
    
    print("=== 🚀 ENTRAÎNEMENT RAPIDE ACCÉLÉRÉ GPU ===")
    print("Pour un entraînement complet, utilisez: python train.py")
    
    # Vérification GPU avant l'entraînement
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU : {gpu_name}")
        if "4060" in gpu_name:
            print("✅ RTX 4060 détectée - Entraînement ultra-rapide !")
        print(f"⚡ 100 épisodes devraient prendre ~1-2 minutes")
    else:
        print("⚠️  Entraînement CPU - Peut prendre 5-10 minutes")
    print("="*45)
    
    # Créer le dossier models
    os.makedirs("models", exist_ok=True)
    
    # Initialiser
    env = SnakeGame(width=400, height=300, block_size=20, speed=50)
    agent = DQNAgent(
        state_size=11,
        action_size=3,
        lr=0.001,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.99  # Décroissance plus rapide pour le test
    )
    
    scores = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, done, _, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.train()
        
        score = info['score']
        scores.append(score)
        
        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
            print(f"Épisode {episode:3d} | Score: {score:2d} | Moyenne: {avg_score:.2f} | ε: {agent.epsilon:.3f}")
    
    env.close()
    
    # Sauvegarder le modèle de test
    agent.save_model("models/snake_dqn_quick_test.pth")
    
    # Statistiques finales
    final_avg = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
    print(f"\n=== RÉSULTATS ===")
    print(f"Score moyen final: {final_avg:.2f}")
    print(f"Meilleur score: {max(scores)}")
    print(f"Modèle sauvegardé: models/snake_dqn_quick_test.pth")
    
    # Graphique simple
    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.6, label='Scores')
    
    # Moyenne mobile
    if len(scores) >= 10:
        moving_avg = []
        for i in range(len(scores)):
            start_idx = max(0, i - 9)
            moving_avg.append(np.mean(scores[start_idx:i+1]))
        plt.plot(moving_avg, color='red', linewidth=2, label='Moyenne mobile (10)')
    
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.title('Entraînement Rapide - Évolution du Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('models/quick_train_results.png', dpi=150)
    plt.show()
    
    return scores

if __name__ == "__main__":
    scores = quick_train()
    
    print("\n🎉 Entraînement rapide terminé!")
    print("\nProchaines étapes:")
    print("1. Pour un entraînement complet: python train.py")
    print("2. Pour tester l'agent: python test.py")
    print("3. Voir le graphique dans: models/quick_train_results.png")
