import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

def save_experiment_config(config, filename="experiments/config.json"):
    """Sauvegarde la configuration d'une expérience."""
    import os
    os.makedirs("experiments", exist_ok=True)
    
    config['timestamp'] = datetime.now().isoformat()
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration sauvegardée: {filename}")

def load_experiment_config(filename="experiments/config.json"):
    """Charge la configuration d'une expérience."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration {filename} non trouvée.")
        return None

def calculate_learning_metrics(scores):
    """
    Calcule diverses métriques d'apprentissage.
    
    Args:
        scores: Liste des scores par épisode
        
    Returns:
        dict: Dictionnaire contenant les métriques
    """
    scores = np.array(scores)
    
    metrics = {
        'mean_score': np.mean(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'std_score': np.std(scores),
        'median_score': np.median(scores),
        'total_episodes': len(scores)
    }
    
    # Calcul de la convergence (derniers 10% vs premiers 10%)
    if len(scores) > 20:
        early_scores = scores[:len(scores)//10]
        late_scores = scores[-len(scores)//10:]
        metrics['early_mean'] = np.mean(early_scores)
        metrics['late_mean'] = np.mean(late_scores)
        metrics['improvement'] = metrics['late_mean'] - metrics['early_mean']
        metrics['improvement_ratio'] = metrics['late_mean'] / max(metrics['early_mean'], 0.1)
    
    # Plateau detection (pas d'amélioration sur les derniers 20%)
    if len(scores) > 100:
        recent_scores = scores[-len(scores)//5:]  # Derniers 20%
        metrics['recent_std'] = np.std(recent_scores)
        metrics['is_plateaued'] = metrics['recent_std'] < 0.5  # Seuil arbitraire
    
    return metrics

def smooth_curve(data, window=50):
    """Lisse une courbe avec une moyenne mobile."""
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        smoothed.append(np.mean(data[start_idx:i+1]))
    
    return smoothed

def plot_comparison(experiments, title="Comparaison d'expériences"):
    """
    Compare plusieurs expériences sur un même graphique.
    
    Args:
        experiments: Dict {nom: scores} pour chaque expérience
        title: Titre du graphique
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (name, scores) in enumerate(experiments.items()):
        color = colors[i % len(colors)]
        
        # Scores bruts (transparent)
        plt.plot(scores, alpha=0.3, color=color)
        
        # Moyenne mobile
        smoothed = smooth_curve(scores, window=100)
        plt.plot(smoothed, label=f"{name} (moyenne)", color=color, linewidth=2)
    
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def hyperparameter_analysis(results_dict):
    """
    Analyse l'impact des hyperparamètres sur les performances.
    
    Args:
        results_dict: {config_name: {'config': config, 'scores': scores}}
    """
    
    # Extraire les métriques pour chaque configuration
    analysis = []
    for name, data in results_dict.items():
        config = data['config']
        scores = data['scores']
        metrics = calculate_learning_metrics(scores)
        
        analysis.append({
            'name': name,
            'lr': config.get('learning_rate', 0.001),
            'gamma': config.get('gamma', 0.9),
            'epsilon_decay': config.get('epsilon_decay', 0.995),
            'batch_size': config.get('batch_size', 32),
            'final_score': metrics['late_mean'] if 'late_mean' in metrics else metrics['mean_score'],
            'max_score': metrics['max_score'],
            'stability': 1 / (metrics['std_score'] + 1)  # Plus la std est faible, plus c'est stable
        })
    
    # Créer des graphiques d'analyse
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Learning rate vs performance
    lrs = [exp['lr'] for exp in analysis]
    scores = [exp['final_score'] for exp in analysis]
    axes[0, 0].scatter(lrs, scores)
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Score Final')
    axes[0, 0].set_title('Impact du Learning Rate')
    axes[0, 0].set_xscale('log')
    
    # Gamma vs performance
    gammas = [exp['gamma'] for exp in analysis]
    axes[0, 1].scatter(gammas, scores)
    axes[0, 1].set_xlabel('Gamma (facteur de discount)')
    axes[0, 1].set_ylabel('Score Final')
    axes[0, 1].set_title('Impact du Gamma')
    
    # Batch size vs performance
    batch_sizes = [exp['batch_size'] for exp in analysis]
    axes[1, 0].scatter(batch_sizes, scores)
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('Score Final')
    axes[1, 0].set_title('Impact de la Taille de Batch')
    
    # Stabilité vs performance
    stabilities = [exp['stability'] for exp in analysis]
    axes[1, 1].scatter(stabilities, scores)
    axes[1, 1].set_xlabel('Stabilité (1/std)')
    axes[1, 1].set_ylabel('Score Final')
    axes[1, 1].set_title('Stabilité vs Performance')
    
    plt.tight_layout()
    plt.show()
    
    # Afficher le tableau des résultats
    print("\n=== ANALYSE DES HYPERPARAMÈTRES ===")
    print(f"{'Nom':<15} {'LR':<8} {'Gamma':<6} {'Decay':<6} {'Batch':<6} {'Score':<8} {'Max':<6} {'Stab':<6}")
    print("-" * 70)
    
    # Trier par score final
    analysis.sort(key=lambda x: x['final_score'], reverse=True)
    
    for exp in analysis:
        print(f"{exp['name']:<15} {exp['lr']:<8.4f} {exp['gamma']:<6.2f} "
              f"{exp['epsilon_decay']:<6.3f} {exp['batch_size']:<6} "
              f"{exp['final_score']:<8.2f} {exp['max_score']:<6} {exp['stability']:<6.3f}")

def reward_shaping_analysis(base_scores, shaped_scores, labels=None):
    """
    Compare l'impact du reward shaping sur l'apprentissage.
    
    Args:
        base_scores: Scores avec récompenses de base
        shaped_scores: Scores avec reward shaping
        labels: Labels pour les différentes stratégies
    """
    
    if labels is None:
        labels = ['Base', 'Shaped']
    
    plt.figure(figsize=(15, 5))
    
    # Courbes d'apprentissage
    plt.subplot(1, 3, 1)
    plt.plot(smooth_curve(base_scores), label=labels[0], linewidth=2)
    plt.plot(smooth_curve(shaped_scores), label=labels[1], linewidth=2)
    plt.xlabel('Épisode')
    plt.ylabel('Score')
    plt.title('Courbes d\'apprentissage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution des scores
    plt.subplot(1, 3, 2)
    plt.hist(base_scores, bins=20, alpha=0.7, label=labels[0])
    plt.hist(shaped_scores, bins=20, alpha=0.7, label=labels[1])
    plt.xlabel('Score')
    plt.ylabel('Fréquence')
    plt.title('Distribution des scores')
    plt.legend()
    
    # Box plot
    plt.subplot(1, 3, 3)
    plt.boxplot([base_scores, shaped_scores], labels=labels)
    plt.ylabel('Score')
    plt.title('Comparaison statistique')
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques
    base_metrics = calculate_learning_metrics(base_scores)
    shaped_metrics = calculate_learning_metrics(shaped_scores)
    
    print("=== ANALYSE DU REWARD SHAPING ===")
    print(f"Méthode de base:")
    print(f"  Score moyen: {base_metrics['mean_score']:.2f}")
    print(f"  Score max: {base_metrics['max_score']}")
    print(f"  Écart-type: {base_metrics['std_score']:.2f}")
    
    print(f"Avec reward shaping:")
    print(f"  Score moyen: {shaped_metrics['mean_score']:.2f}")
    print(f"  Score max: {shaped_metrics['max_score']}")
    print(f"  Écart-type: {shaped_metrics['std_score']:.2f}")
    
    improvement = (shaped_metrics['mean_score'] - base_metrics['mean_score']) / base_metrics['mean_score'] * 100
    print(f"Amélioration: {improvement:.1f}%")

def export_results_to_csv(scores, filename="results.csv"):
    """Exporte les résultats vers un fichier CSV."""
    
    # Version simple sans pandas
    with open(filename, 'w') as f:
        f.write("episode,score,moving_avg_50,moving_avg_100,moving_avg_200\n")
        
        for i, score in enumerate(scores):
            # Calculer les moyennes mobiles
            avg_50 = np.mean(scores[max(0, i-49):i+1])
            avg_100 = np.mean(scores[max(0, i-99):i+1])
            avg_200 = np.mean(scores[max(0, i-199):i+1])
            
            f.write(f"{i},{score},{avg_50:.3f},{avg_100:.3f},{avg_200:.3f}\n")
    
    print(f"Résultats exportés vers {filename}")

def generate_report(experiment_name, scores, config, model_path):
    """
    Génère un rapport complet d'expérience.
    
    Args:
        experiment_name: Nom de l'expérience
        scores: Scores obtenus
        config: Configuration utilisée
        model_path: Chemin du modèle sauvegardé
    """
    
    metrics = calculate_learning_metrics(scores)
    
    report = f"""
# Rapport d'Expérience: {experiment_name}

## Configuration
- **Épisodes d'entraînement**: {len(scores)}
- **Learning Rate**: {config.get('learning_rate', 'N/A')}
- **Gamma**: {config.get('gamma', 'N/A')}
- **Epsilon Decay**: {config.get('epsilon_decay', 'N/A')}
- **Batch Size**: {config.get('batch_size', 'N/A')}
- **Architecture**: {config.get('architecture', 'N/A')}

## Résultats
- **Score moyen**: {metrics['mean_score']:.2f}
- **Score maximum**: {metrics['max_score']}
- **Score minimum**: {metrics['min_score']}
- **Écart-type**: {metrics['std_score']:.2f}
- **Score médian**: {metrics['median_score']:.2f}

## Convergence
- **Score initial (10% premiers épisodes)**: {metrics.get('early_mean', 'N/A'):.2f}
- **Score final (10% derniers épisodes)**: {metrics.get('late_mean', 'N/A'):.2f}
- **Amélioration**: {metrics.get('improvement', 'N/A'):.2f}
- **Ratio d'amélioration**: {metrics.get('improvement_ratio', 'N/A'):.2f}x

## Modèle
- **Chemin**: {model_path}
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Notes
Agent entraîné avec succès. Performance {'stable' if metrics.get('is_plateaued', False) else 'en amélioration'}.
"""
    
    # Sauvegarder le rapport
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Rapport généré: {report_path}")
    return report_path
