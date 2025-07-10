import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Tuple, List

class DQN(nn.Module):
    """
    Deep Q-Network pour l'agent Snake.
    
    Architecture simple:
    - Couche d'entrée : état du jeu (11 features)
    - Couches cachées : 256 et 256 neurones avec ReLU
    - Couche de sortie : Q-values pour chaque action (3 actions)
    """
    
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ReplayBuffer:
    """
    Buffer de replay pour stocker les expériences de l'agent.
    Implémente l'échantillonnage aléatoire pour stabiliser l'apprentissage.
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Ajoute une expérience au buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Échantillonne un batch d'expériences."""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in experiences])
        actions = torch.LongTensor([e[1] for e in experiences])
        rewards = torch.FloatTensor([e[2] for e in experiences])
        next_states = torch.FloatTensor([e[3] for e in experiences])
        dones = torch.BoolTensor([e[4] for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Agent DQN pour jouer à Snake.
    
    Implémente:
    - ε-greedy policy pour l'exploration
    - Target network pour la stabilité
    - Experience replay
    - Double DQN (optionnel)
    """
    
    def __init__(self, state_size=11, action_size=3, lr=0.001, gamma=0.9, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update=1000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma  # Facteur de discount
        self.epsilon = epsilon  # Probabilité d'exploration
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device (CPU/GPU) - Optimisé pour RTX 4060
        self.device = self._setup_device()
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(self.device)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f} GB")
        
        # Réseaux de neurones
        self.q_network = DQN(state_size, 256, action_size).to(self.device)
        self.target_network = DQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialiser target network avec les poids du main network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Compteurs
        self.step_count = 0
    
    def _setup_device(self):
        """
        Configure le device optimal pour l'entraînement.
        Priorise le GPU NVIDIA RTX 4060 si disponible.
        """
        if not torch.cuda.is_available():
            print("❌ CUDA non disponible, utilisation du CPU")
            return torch.device("cpu")
        
        gpu_count = torch.cuda.device_count()
        print(f"🔍 {gpu_count} GPU(s) détecté(s)")
        
        # Rechercher spécifiquement la RTX 4060
        target_gpu = None
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"   GPU {i}: {gpu_name}")
            
            # Cibler spécifiquement la RTX 4060
            if "4060" in gpu_name or "RTX 4060" in gpu_name:
                target_gpu = i
                print(f"✅ RTX 4060 trouvée sur GPU {i}")
                break
        
        # Si RTX 4060 trouvée, l'utiliser, sinon utiliser GPU 0
        if target_gpu is not None:
            device = torch.device(f"cuda:{target_gpu}")
            print(f"🚀 Utilisation de la RTX 4060 (GPU {target_gpu})")
        else:
            device = torch.device("cuda:0")  # GPU par défaut
            print(f"⚠️ RTX 4060 non trouvée, utilisation du GPU par défaut")
        
        # Optimisations CUDA pour RTX 4060
        torch.backends.cudnn.benchmark = True  # Optimise les convolutions
        torch.backends.cudnn.deterministic = False  # Performance vs reproductibilité
        
        return device
        
    def get_action(self, state, training=True):
        """
        Sélectionne une action selon la politique ε-greedy.
        
        Args:
            state: État actuel du jeu
            training: Si False, utilise seulement l'exploitation (pas d'exploration)
        """
        if training and random.random() < self.epsilon:
            # Exploration: action aléatoire
            return random.choice(range(self.action_size))
        
        # Exploitation: action avec la plus haute Q-value
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience dans le replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        """
        Entraîne l'agent sur un batch d'expériences.
        Implémente l'algorithme DQN avec target network.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Échantillonner un batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Q-values actuelles
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Q-values cibles (utilise target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calcul de la loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour la stabilité
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Mise à jour du target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Décroissance d'epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Sauvegarde le modèle."""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
        print(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath):
        """Charge un modèle sauvegardé."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.step_count = checkpoint.get('step_count', 0)
        print(f"Modèle chargé: {filepath}")
    
    def get_model_info(self):
        """Retourne des informations sur le modèle."""
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'device': str(self.device)
        }
