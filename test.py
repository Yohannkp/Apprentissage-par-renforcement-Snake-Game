import pygame
import numpy as np
import matplotlib.pyplot as plt
from env.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
import time
import os

def test_agent(model_path="models/snake_dqn_best.pth", n_games=5, render=True, speed=10):
    """
    Teste l'agent entraîné et affiche ses performances.
    
    Args:
        model_path: Chemin vers le modèle entraîné
        n_games: Nombre de parties à jouer
        render: Si True, affiche le jeu en temps réel
        speed: Vitesse du jeu (plus élevé = plus rapide)
    """
    
    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle {model_path} n'existe pas.")
        print("Veuillez d'abord entraîner l'agent avec train.py")
        return
    
    # Initialiser l'environnement et l'agent
    env = SnakeGame(width=640, height=480, block_size=20, speed=speed)
    agent = DQNAgent()
    
    # Charger le modèle entraîné
    agent.load_model(model_path)
    print(f"Modèle chargé: {model_path}")
    print(f"Information du modèle: {agent.get_model_info()}")
    
    # Statistiques de test
    scores = []
    total_steps = []
    
    print(f"\n=== TEST DE L'AGENT SUR {n_games} PARTIES ===")
    
    for game in range(n_games):
        state, _ = env.reset()
        done = False
        steps = 0
        
        print(f"\nPartie {game + 1}/{n_games}")
        
        while not done:
            # Action de l'agent (pas d'exploration, seulement exploitation)
            action = agent.get_action(state, training=False)
            
            # Exécuter l'action
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            steps += 1
            
            # Afficher le jeu
            if render:
                env.render()
                
                # Gérer les événements pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
            
            # Pause pour visualiser
            if render:
                time.sleep(0.1)
        
        # Enregistrer les statistiques
        score = info['score']
        scores.append(score)
        total_steps.append(steps)
        
        print(f"Score: {score}, Étapes: {steps}")
        
        if render:
            # Pause entre les parties
            time.sleep(2)
    
    # Fermer l'environnement
    env.close()
    
    # Afficher les statistiques finales
    print(f"\n=== RÉSULTATS FINAUX ===")
    print(f"Score moyen: {np.mean(scores):.2f}")
    print(f"Score maximum: {max(scores)}")
    print(f"Score minimum: {min(scores)}")
    print(f"Écart-type: {np.std(scores):.2f}")
    print(f"Étapes moyennes par partie: {np.mean(total_steps):.2f}")
    
    return scores, total_steps

def compare_random_vs_trained(model_path="models/snake_dqn_best.pth", n_games=20):
    """
    Compare les performances d'un agent aléatoire vs l'agent entraîné.
    
    Args:
        model_path: Chemin vers le modèle entraîné
        n_games: Nombre de parties pour la comparaison
    """
    
    env = SnakeGame(width=640, height=480, block_size=20, speed=100)  # Vitesse élevée pour les tests
    
    # Test agent aléatoire
    print("=== TEST AGENT ALÉATOIRE ===")
    random_scores = []
    
    for game in range(n_games):
        state, _ = env.reset()
        done = False
        
        while not done:
            # Action aléatoire
            action = np.random.randint(0, 3)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
        
        random_scores.append(info['score'])
    
    # Test agent entraîné
    print("=== TEST AGENT ENTRAÎNÉ ===")
    if os.path.exists(model_path):
        agent = DQNAgent()
        agent.load_model(model_path)
        trained_scores = []
        
        for game in range(n_games):
            state, _ = env.reset()
            done = False
            
            while not done:
                # Action de l'agent entraîné
                action = agent.get_action(state, training=False)
                next_state, reward, done, _, info = env.step(action)
                state = next_state
            
            trained_scores.append(info['score'])
    else:
        print(f"Modèle {model_path} non trouvé!")
        trained_scores = [0] * n_games
    
    env.close()
    
    # Afficher la comparaison
    print(f"\n=== COMPARAISON ===")
    print(f"Agent aléatoire - Score moyen: {np.mean(random_scores):.2f}")
    print(f"Agent entraîné - Score moyen: {np.mean(trained_scores):.2f}")
    print(f"Amélioration: {np.mean(trained_scores) / max(np.mean(random_scores), 0.1):.2f}x")
    
    # Graphique de comparaison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(random_scores, bins=10, alpha=0.7, label='Agent aléatoire', color='red')
    plt.hist(trained_scores, bins=10, alpha=0.7, label='Agent entraîné', color='blue')
    plt.xlabel('Score')
    plt.ylabel('Fréquence')
    plt.title('Distribution des scores')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.boxplot([random_scores, trained_scores], labels=['Aléatoire', 'Entraîné'])
    plt.ylabel('Score')
    plt.title('Comparaison des performances')
    
    plt.tight_layout()
    plt.savefig('models/comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return random_scores, trained_scores

def play_interactive(model_path="models/snake_dqn_best.pth"):
    """
    Mode interactif où l'utilisateur peut voir l'agent jouer en continu.
    Appuyez sur SPACE pour démarrer une nouvelle partie, ESC pour quitter.
    """
    
    if not os.path.exists(model_path):
        print(f"Modèle {model_path} non trouvé!")
        return
    
    # Initialiser
    env = SnakeGame(width=640, height=480, block_size=20, speed=15)
    agent = DQNAgent()
    agent.load_model(model_path)
    
    print("=== MODE INTERACTIF ===")
    print("SPACE: Nouvelle partie")
    print("ESC: Quitter")
    
    running = True
    game_active = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Nouvelle partie
                    state, _ = env.reset()
                    game_active = True
                    print("Nouvelle partie démarrée!")
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        if game_active:
            # Action de l'agent
            action = agent.get_action(state, training=False)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            
            if done:
                print(f"Partie terminée! Score: {info['score']}")
                game_active = False
        
        # Affichage
        env.render()
        time.sleep(0.1)
    
    env.close()

def analyze_training_progress():
    """
    Analyse et affiche les résultats d'entraînement sauvegardés.
    """
    
    try:
        scores = np.load("models/training_scores.npy")
        mean_scores = np.load("models/training_mean_scores.npy")
    except FileNotFoundError:
        print("Aucune donnée d'entraînement trouvée.")
        print("Veuillez d'abord entraîner l'agent avec train.py")
        return
    
    print(f"=== ANALYSE DES RÉSULTATS D'ENTRAÎNEMENT ===")
    print(f"Nombre total d'épisodes: {len(scores)}")
    print(f"Score final moyen: {mean_scores[-1]:.2f}")
    print(f"Meilleur score atteint: {max(scores)}")
    print(f"Score à l'épisode 100: {mean_scores[99] if len(mean_scores) > 99 else 'N/A'}")
    print(f"Score à l'épisode 500: {mean_scores[499] if len(mean_scores) > 499 else 'N/A'}")
    print(f"Score à l'épisode 1000: {mean_scores[999] if len(mean_scores) > 999 else 'N/A'}")
    
    # Graphique détaillé
    plt.figure(figsize=(15, 10))
    
    # Scores bruts et moyenne
    plt.subplot(2, 2, 1)
    plt.plot(scores, alpha=0.3, color='lightblue')
    plt.plot(mean_scores, color='red', linewidth=2)
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.title('Évolution complète du score')
    plt.grid(True, alpha=0.3)
    
    # Zoom sur les 200 premiers épisodes
    plt.subplot(2, 2, 2)
    plt.plot(scores[:200], alpha=0.6, color='blue')
    plt.plot(mean_scores[:200], color='red', linewidth=2)
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.title('Premiers 200 épisodes')
    plt.grid(True, alpha=0.3)
    
    # Distribution des scores
    plt.subplot(2, 2, 3)
    plt.hist(scores, bins=50, alpha=0.7, color='green')
    plt.xlabel('Score')
    plt.ylabel('Fréquence')
    plt.title('Distribution des scores')
    plt.grid(True, alpha=0.3)
    
    # Moyenne mobile
    plt.subplot(2, 2, 4)
    if len(scores) >= 100:
        moving_avg = [np.mean(scores[max(0, i-99):i+1]) for i in range(len(scores))]
        plt.plot(moving_avg, color='purple', linewidth=2)
    plt.xlabel('Épisode')
    plt.ylabel('Score moyen (100 épisodes)')
    plt.title('Moyenne mobile')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("=== MENU DE TEST ===")
    print("1. Tester l'agent (visuel)")
    print("2. Comparer aléatoire vs entraîné")
    print("3. Mode interactif")
    print("4. Analyser les résultats d'entraînement")
    
    choice = input("Choisissez une option (1-4): ")
    
    if choice == "1":
        test_agent(render=True, n_games=3)
    elif choice == "2":
        compare_random_vs_trained()
    elif choice == "3":
        play_interactive()
    elif choice == "4":
        analyze_training_progress()
    else:
        print("Option invalide!")
