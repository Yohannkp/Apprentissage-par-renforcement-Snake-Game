#!/usr/bin/env python3
"""
Mode de jeu manuel pour Snake.
Contrôles:
- Flèches directionnelles : Déplacer le serpent
- ESPACE : Redémarrer la partie
- ESC : Quitter
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pygame
import time
from env.snake_game import SnakeGame, Direction

class ManualSnakeGame:
    """Version manuelle du jeu Snake pour jouer soi-même."""
    
    def __init__(self, width=640, height=480, block_size=20, speed=10):
        pygame.init()
        
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed
        
        # Couleurs
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        
        # Interface
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake - Mode Manuel')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)
        
        # État du jeu
        self.reset_game()
        
    def reset_game(self):
        """Réinitialise le jeu."""
        self.head = [self.width // 2, self.height // 2]
        self.snake = [
            self.head.copy(),
            [self.head[0] - self.block_size, self.head[1]],
            [self.head[0] - 2 * self.block_size, self.head[1]]
        ]
        self.direction = Direction.RIGHT
        self.score = 0
        self.game_over = False
        self.place_food()
        
    def place_food(self):
        """Place la nourriture aléatoirement."""
        import random
        while True:
            x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
            y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
            self.food = [x, y]
            if self.food not in self.snake:
                break
    
    def handle_input(self):
        """Gère les entrées clavier."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                if event.key == pygame.K_SPACE:
                    self.reset_game()
                
                # Contrôles directionnels (empêche les retours en arrière)
                if not self.game_over:
                    if event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                    elif event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                    elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                        self.direction = Direction.DOWN
        
        return True
    
    def move_snake(self):
        """Déplace le serpent."""
        if self.game_over:
            return
        
        # Calculer la nouvelle position de la tête
        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size
        
        self.head = [x, y]
        
        # Vérifier les collisions
        if self.check_collision():
            self.game_over = True
            return
        
        # Ajouter la nouvelle tête
        self.snake.insert(0, self.head.copy())
        
        # Vérifier si on mange la nourriture
        if self.head == self.food:
            self.score += 1
            self.place_food()
        else:
            # Retirer la queue si on n'a pas mangé
            self.snake.pop()
    
    def check_collision(self):
        """Vérifie les collisions."""
        # Collision avec les bords
        if (self.head[0] >= self.width or self.head[0] < 0 or
            self.head[1] >= self.height or self.head[1] < 0):
            return True
        
        # Collision avec soi-même
        if self.head in self.snake:
            return True
        
        return False
    
    def draw(self):
        """Dessine le jeu."""
        self.display.fill(self.BLACK)
        
        # Dessiner le serpent
        for i, segment in enumerate(self.snake):
            if i == 0:  # Tête
                pygame.draw.rect(self.display, self.YELLOW,
                               pygame.Rect(segment[0], segment[1], self.block_size, self.block_size))
                pygame.draw.rect(self.display, self.GREEN,
                               pygame.Rect(segment[0] + 2, segment[1] + 2, self.block_size - 4, self.block_size - 4))
            else:  # Corps
                pygame.draw.rect(self.display, self.GREEN,
                               pygame.Rect(segment[0], segment[1], self.block_size, self.block_size))
                pygame.draw.rect(self.display, self.BLUE,
                               pygame.Rect(segment[0] + 4, segment[1] + 4, self.block_size - 8, self.block_size - 8))
        
        # Dessiner la nourriture
        pygame.draw.rect(self.display, self.RED,
                        pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        
        # Afficher le score
        score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.display.blit(score_text, [10, 10])
        
        # Afficher la longueur du serpent
        length_text = self.font.render(f"Longueur: {len(self.snake)}", True, self.WHITE)
        self.display.blit(length_text, [10, 50])
        
        # Instructions
        if not self.game_over:
            controls_text = pygame.font.Font(None, 24).render("Flèches: Déplacer | ESPACE: Restart | ESC: Quitter", True, self.WHITE)
            text_rect = controls_text.get_rect()
            text_rect.centerx = self.width // 2
            self.display.blit(controls_text, [text_rect.x, self.height - 30])
        
        # Message Game Over
        if self.game_over:
            game_over_text = self.big_font.render("GAME OVER", True, self.RED)
            text_rect = game_over_text.get_rect()
            text_rect.center = (self.width // 2, self.height // 2 - 50)
            self.display.blit(game_over_text, text_rect)
            
            restart_text = self.font.render("Appuyez sur ESPACE pour recommencer", True, self.WHITE)
            text_rect = restart_text.get_rect()
            text_rect.center = (self.width // 2, self.height // 2 + 20)
            self.display.blit(restart_text, text_rect)
            
            final_score_text = self.font.render(f"Score final: {self.score}", True, self.YELLOW)
            text_rect = final_score_text.get_rect()
            text_rect.center = (self.width // 2, self.height // 2 + 60)
            self.display.blit(final_score_text, text_rect)
        
        pygame.display.flip()
    
    def run(self):
        """Boucle principale du jeu."""
        print("=== SNAKE - MODE MANUEL ===")
        print("Contrôles:")
        print("  ↑↓←→  : Déplacer le serpent")
        print("  ESPACE : Redémarrer")
        print("  ESC    : Quitter")
        print("\nBut: Mangez les pommes rouges pour grandir!")
        
        running = True
        while running:
            # Gérer les entrées
            running = self.handle_input()
            
            # Déplacer le serpent
            self.move_snake()
            
            # Dessiner
            self.draw()
            
            # Contrôler la vitesse
            self.clock.tick(self.speed)
        
        pygame.quit()
        print(f"\nMerci d'avoir joué! Score final: {self.score}")

def main():
    """Fonction principale."""
    print("Choisissez le niveau de difficulté:")
    print("1. Facile (vitesse lente)")
    print("2. Normal (vitesse moyenne)")
    print("3. Difficile (vitesse rapide)")
    
    try:
        choice = input("Votre choix (1-3): ").strip()
        
        if choice == "1":
            speed = 6
            print("Mode Facile sélectionné")
        elif choice == "2":
            speed = 10
            print("Mode Normal sélectionné")
        elif choice == "3":
            speed = 15
            print("Mode Difficile sélectionné")
        else:
            speed = 10
            print("Mode Normal par défaut")
        
        # Démarrer le jeu
        game = ManualSnakeGame(width=640, height=480, block_size=20, speed=speed)
        game.run()
        
    except KeyboardInterrupt:
        print("\nJeu interrompu par l'utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
