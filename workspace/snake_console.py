"""
Console-based Snake Game for Windows
Uses msvcrt for non-blocking keyboard input
"""

import os
import sys
import time
import random
import msvcrt

class ConsoleSnake:
    def __init__(self, width=40, height=20):
        self.width = width
        self.height = height
        self.reset_game()
        
    def reset_game(self):
        """Reset game state"""
        # Snake starts in the middle
        self.snake = [
            (self.width // 2, self.height // 2),
            (self.width // 2 - 1, self.height // 2),
            (self.width // 2 - 2, self.height // 2)
        ]
        self.direction = "right"
        self.next_direction = "right"
        self.score = 0
        self.game_over = False
        self.food = self.generate_food()
        self.speed = 0.1  # seconds between updates
        
    def generate_food(self):
        """Generate food at random position"""
        while True:
            food = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1)
            )
            if food not in self.snake:
                return food
    
    def get_input(self):
        """Get keyboard input without blocking"""
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if key == 'w' or key == 'à':
                self.next_direction = "up"
            elif key == 's' or key == 'ò':
                self.next_direction = "down"
            elif key == 'a' or key == 'æ':
                self.next_direction = "left"
            elif key == 'd' or key == 'ô':
                self.next_direction = "right"
            elif key == 'q':
                return False
            elif key == 'r' and self.game_over:
                self.reset_game()
        return True
    
    def update(self):
        """Update game state"""
        if self.game_over:
            return
        
        # Update direction
        opposites = {
            "left": "right",
            "right": "left",
            "up": "down",
            "down": "up"
        }
        if opposites.get(self.next_direction) != self.direction:
            self.direction = self.next_direction
        
        # Calculate new head
        head_x, head_y = self.snake[0]
        if self.direction == "left":
            new_head = (head_x - 1, head_y)
        elif self.direction == "right":
            new_head = (head_x + 1, head_y)
        elif self.direction == "up":
            new_head = (head_x, head_y - 1)
        elif self.direction == "down":
            new_head = (head_x, head_y + 1)
        
        # Check collisions
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.game_over = True
            return
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check food
        if new_head == self.food:
            self.score += 10
            self.food = self.generate_food()
            # Increase speed every 5 foods
            if self.score % 50 == 0 and self.speed > 0.05:
                self.speed -= 0.01
        else:
            self.snake.pop()
    
    def draw(self):
        """Draw the game board"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Create empty board
        board = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            if 0 <= x < self.width and 0 <= y < self.height:
                board[y][x] = '●' if i == 0 else '○'
        
        # Draw food
        fx, fy = self.food
        if 0 <= fx < self.width and 0 <= fy < self.height:
            board[fy][fx] = '★'
        
        # Print board
        print('┌' + '─' * self.width + '┐')
        for row in board:
            print('│' + ''.join(row) + '│')
        print('└' + '─' * self.width + '┘')
        
        # Print info
        print(f"Score: {self.score} | Speed: {1/self.speed:.1f} FPS")
        print("Controls: W=Up, A=Left, S=Down, D=Right, Q=Quit")
        if self.game_over:
            print("\n" + "═" * 40)
            print("GAME OVER!")
            print(f"Final Score: {self.score}")
            print("Press R to restart or Q to quit")
            print("═" * 40)
    
    def run(self):
        """Main game loop"""
        print("Console Snake Game")
        print("Press any key to start...")
        
        while True:
            # Handle input
            if not self.get_input():
                break
            
            # Update game state
            self.update()
            
            # Draw
            self.draw()
            
            # Control game speed
            if not self.game_over:
                time.sleep(self.speed)
            else:
                time.sleep(0.1)

if __name__ == "__main__":
    try:
        game = ConsoleSnake(40, 20)
        game.run()
    except KeyboardInterrupt:
        print("\nGame exited.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you're running on Windows with proper console.")