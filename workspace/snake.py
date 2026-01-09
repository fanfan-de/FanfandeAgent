"""
Snake Game using tkinter
Controls: Arrow keys to change direction
"""

import tkinter as tk
import random
import time

class SnakeGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Snake Game")
        self.master.resizable(False, False)
        
        # Game constants
        self.cell_size = 20
        self.width = 30
        self.height = 20
        self.speed = 100  # milliseconds between updates
        
        # Create canvas
        self.canvas = tk.Canvas(
            master, 
            width=self.width * self.cell_size,
            height=self.height * self.cell_size,
            bg="black"
        )
        self.canvas.pack()
        
        # Initialize game state
        self.reset_game()
        
        # Bind arrow keys
        self.master.bind("<KeyPress-Left>", lambda e: self.change_direction("left"))
        self.master.bind("<KeyPress-Right>", lambda e: self.change_direction("right"))
        self.master.bind("<KeyPress-Up>", lambda e: self.change_direction("up"))
        self.master.bind("<KeyPress-Down>", lambda e: self.change_direction("down"))
        
        # Start the game
        self.master.after(self.speed, self.update)
        
    def reset_game(self):
        """Reset the game to initial state"""
        # Snake starts as 3 segments in the middle
        self.snake = [
            (self.width // 2, self.height // 2),
            (self.width // 2 - 1, self.height // 2),
            (self.width // 2 - 2, self.height // 2)
        ]
        self.direction = "right"
        self.next_direction = "right"
        self.score = 0
        self.game_over = False
        
        # Generate first food
        self.food = self.generate_food()
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw initial state
        self.draw()
        
    def generate_food(self):
        """Generate food at a random position not occupied by snake"""
        while True:
            food = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1)
            )
            if food not in self.snake:
                return food
    
    def change_direction(self, new_direction):
        """Change direction if not opposite to current direction"""
        opposites = {
            "left": "right",
            "right": "left",
            "up": "down",
            "down": "up"
        }
        if opposites.get(new_direction) != self.direction:
            self.next_direction = new_direction
    
    def update(self):
        """Update game state"""
        if self.game_over:
            return
            
        # Update direction
        self.direction = self.next_direction
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        if self.direction == "left":
            new_head = (head_x - 1, head_y)
        elif self.direction == "right":
            new_head = (head_x + 1, head_y)
        elif self.direction == "up":
            new_head = (head_x, head_y - 1)
        elif self.direction == "down":
            new_head = (head_x, head_y + 1)
        
        # Check for collisions
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.game_over = True
            self.draw_game_over()
            return
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 10
            self.food = self.generate_food()
            # Increase speed slightly every 5 foods
            if self.score % 50 == 0 and self.speed > 50:
                self.speed -= 10
        else:
            # Remove tail if no food eaten
            self.snake.pop()
        
        # Draw updated state
        self.draw()
        
        # Schedule next update
        self.master.after(self.speed, self.update)
    
    def draw(self):
        """Draw the current game state"""
        self.canvas.delete("all")
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = "lime green" if i == 0 else "green"  # Head is brighter
            self.draw_cell(x, y, color)
        
        # Draw food
        self.draw_cell(self.food[0], self.food[1], "red")
        
        # Draw score
        self.canvas.create_text(
            10, 10,
            text=f"Score: {self.score}",
            anchor="nw",
            fill="white",
            font=("Arial", 12)
        )
    
    def draw_cell(self, x, y, color):
        """Draw a cell at grid coordinates"""
        x1 = x * self.cell_size
        y1 = y * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
    
    def draw_game_over(self):
        """Draw game over message"""
        self.canvas.create_text(
            self.width * self.cell_size // 2,
            self.height * self.cell_size // 2 - 20,
            text="GAME OVER",
            fill="red",
            font=("Arial", 24, "bold")
        )
        self.canvas.create_text(
            self.width * self.cell_size // 2,
            self.height * self.cell_size // 2 + 20,
            text=f"Final Score: {self.score}",
            fill="white",
            font=("Arial", 16)
        )
        self.canvas.create_text(
            self.width * self.cell_size // 2,
            self.height * self.cell_size // 2 + 50,
            text="Press R to restart",
            fill="yellow",
            font=("Arial", 14)
        )
        self.master.bind("r", lambda e: self.reset_game())
        self.master.bind("R", lambda e: self.reset_game())

def main():
    root = tk.Tk()
    game = SnakeGame(root)
    root.mainloop()

if __name__ == "__main__":
    main()