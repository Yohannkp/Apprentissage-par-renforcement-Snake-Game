import os
import numpy as np
import matplotlib.pyplot as plt
from env.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
from tqdm import tqdm
import torch

def train_agent(n_episodes=2000, save_interval=100, model_path="models/snake_dqn.pth"):
    """
    Entraîne l'agent DQN sur le jeu Snake.
    
    Args:
        n_episodes: Nombre d'épisodes d'entraînement
        save_interval: Intervalle de sauvegarde du modèle
        model_path: Chemin de sauvegarde du modèle
    """
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs("models", exist_ok=True)
    
    # Initialiser l'environnement et l'agent
    env = SnakeGame(width=640, height=480, block_size=20, speed=50)
    agent = DQNAgent(
        state_size=11,
        action_size=3,
        lr=0.001,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update=1000
    )
    
    # Métriques d'entraînement
    scores = []
    mean_scores = []
    total_score = 0
    best_score = 0
    
    print("🚀 DÉBUT DE L'ENTRAÎNEMENT ACCÉLÉRÉ GPU")
    print("="*50)
    
    # Afficher les informations GPU avant l'entraînement
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 GPU utilisé : {gpu_name}")
        print(f"💾 Mémoire GPU : {gpu_memory:.1f} GB")
        
        # Vérifier si c'est la RTX 4060
        if "4060" in gpu_name:
            print("✅ RTX 4060 détectée - Performance optimale !")
        
        print(f"⚡ Entraînement accéléré GPU activé")
    else:
        print("⚠️  GPU non disponible - Utilisation CPU (plus lent)")
    
    print(f"📊 Information du modèle: {agent.get_model_info()}")
    print("="*50)
    
    for episode in tqdm(range(n_episodes), desc="Entraînement"):
        # Réinitialiser l'environnement
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Choisir une action
            action = agent.get_action(state, training=True)
            
            # Exécuter l'action
            next_state, reward, done, _, info = env.step(action)
            
            # Stocker l'expérience
            agent.remember(state, action, reward, next_state, done)
            
            # Passer à l'état suivant
            state = next_state
            total_reward += reward
            
            # Entraîner l'agent
            agent.train()
        
        # Statistiques
        score = info['score']
        scores.append(score)
        total_score += score
        mean_score = total_score / (episode + 1)
        mean_scores.append(mean_score)
        
        # Mise à jour du meilleur score
        if score > best_score:
            best_score = score
            # Sauvegarder le meilleur modèle
            agent.save_model(model_path.replace('.pth', '_best.pth'))
        
        # Affichage périodique
        if episode % 100 == 0:
            print(f"\nÉpisode {episode}")
            print(f"Score: {score}, Score moyen: {mean_score:.2f}")
            print(f"Meilleur score: {best_score}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Taille buffer: {len(agent.memory)}")
        
        # Sauvegarde périodique
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(model_path.replace('.pth', f'_episode_{episode}.pth'))
    
    # Sauvegarde finale
    agent.save_model(model_path)
    
    # Fermer l'environnement
    env.close()
    
    # Afficher les résultats finaux
    print(f"\nEntraînement terminé!")
    print(f"Score final moyen: {mean_score:.2f}")
    print(f"Meilleur score atteint: {best_score}")
    
    # Sauvegarder les métriques
    np.save("models/training_scores.npy", scores)
    np.save("models/training_mean_scores.npy", mean_scores)
    
    return scores, mean_scores, agent

def plot_training_results(scores, mean_scores, save_path="models/training_plot.png"):
    """
    Affiche et sauvegarde les résultats d'entraînement.
    
    Args:
        scores: Liste des scores par épisode
        mean_scores: Liste des scores moyens
        save_path: Chemin de sauvegarde du graphique
    """
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Scores par épisode
    plt.subplot(2, 1, 1)
    plt.plot(scores, alpha=0.6, color='blue', label='Score par épisode')
    plt.plot(mean_scores, color='red', linewidth=2, label='Score moyen')
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.title('Évolution du score pendant l\'entraînement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Moyenne mobile des 100 derniers épisodes
    plt.subplot(2, 1, 2)
    if len(scores) >= 100:
        moving_avg = []
        for i in range(len(scores)):
            if i >= 99:
                moving_avg.append(np.mean(scores[i-99:i+1]))
            else:
                moving_avg.append(np.mean(scores[:i+1]))
        plt.plot(moving_avg, color='green', linewidth=2, label='Moyenne mobile (100 épisodes)')
    else:
        plt.plot(mean_scores, color='green', linewidth=2, label='Score moyen')
    
    plt.xlabel('Épisode')
    plt.ylabel('Score moyen')
    plt.title('Moyenne mobile des scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Graphique sauvegardé: {save_path}")

def resume_training(model_path, additional_episodes=1000):
    """
    Reprend l'entraînement à partir d'un modèle sauvegardé.
    
    Args:
        model_path: Chemin vers le modèle à charger
        additional_episodes: Nombre d'épisodes supplémentaires
    """
    # Charger les données précédentes si elles existent
    try:
        previous_scores = np.load("models/training_scores.npy").tolist()
        previous_mean_scores = np.load("models/training_mean_scores.npy").tolist()
        print(f"Chargement des métriques précédentes: {len(previous_scores)} épisodes")
    except FileNotFoundError:
        previous_scores = []
        previous_mean_scores = []
        print("Aucune métrique précédente trouvée, démarrage à zéro")
    
    # Initialiser l'environnement et l'agent
    env = SnakeGame()
    agent = DQNAgent()
    
    # Charger le modèle
    agent.load_model(model_path)
    
    # Continuer l'entraînement
    print(f"Reprise de l'entraînement pour {additional_episodes} épisodes supplémentaires...")
    
    new_scores, new_mean_scores, _ = train_agent(
        n_episodes=additional_episodes,
        save_interval=100,
        model_path=model_path
    )
    
    # Combiner avec les résultats précédents
    all_scores = previous_scores + new_scores
    all_mean_scores = previous_mean_scores + new_mean_scores
    
    # Afficher les résultats
    plot_training_results(all_scores, all_mean_scores)
    
    return all_scores, all_mean_scores

if __name__ == "__main__":
    # Configuration d'entraînement
    EPISODES = 2000
    MODEL_PATH = "models/snake_dqn.pth"
    
    print("=== ENTRAÎNEMENT DE L'AGENT SNAKE DQN ===")
    print(f"Nombre d'épisodes: {EPISODES}")
    print(f"Chemin du modèle: {MODEL_PATH}")
    
    # Entraîner l'agent
    scores, mean_scores, trained_agent = train_agent(
        n_episodes=EPISODES,
        save_interval=200,
        model_path=MODEL_PATH
    )
    
    # Afficher les résultats
    plot_training_results(scores, mean_scores)
    
    print("\nEntraînement terminé! Utilisez test.py pour voir l'agent jouer.")
