"""
简单的贪吃蛇游戏
使用 Python 的 turtle 模块
控制：上下左右箭头键
"""
import turtle
import time
import random

# 游戏设置
WIDTH = 600
HEIGHT = 600
GRID_SIZE = 20
DELAY = 0.1  # 控制游戏速度

class SnakeGame:
    def __init__(self):
        # 初始化屏幕
        self.screen = turtle.Screen()
        self.screen.setup(WIDTH, HEIGHT)
        self.screen.title("贪吃蛇游戏")
        self.screen.bgcolor("black")
        self.screen.tracer(0)  # 关闭自动刷新
        
        # 初始化蛇
        self.snake = []
        self.create_snake()
        self.head = self.snake[0]
        
        # 初始化食物
        self.food = turtle.Turtle()
        self.food.shape("circle")
        self.food.color("red")
        self.food.penup()
        self.food.speed(0)
        
        # 分数
        self.score = 0
        self.score_display = turtle.Turtle()
        self.score_display.speed(0)
        self.score_display.color("white")
        self.score_display.penup()
        self.score_display.hideturtle()
        self.score_display.goto(0, HEIGHT//2 - 40)
        
        # 方向
        self.direction = "right"
        self.next_direction = "right"
        
        # 游戏状态
        self.game_over = False
        
        # 设置键盘监听
        self.setup_keyboard()
        
        # 生成第一个食物
        self.spawn_food()
        self.update_score()
    
    def create_snake(self):
        """创建初始的蛇"""
        for i in range(3):
            segment = turtle.Turtle()
            segment.shape("square")
            segment.color("green")
            segment.penup()
            segment.speed(0)
            segment.goto(-i * GRID_SIZE, 0)
            self.snake.append(segment)
    
    def setup_keyboard(self):
        """设置键盘控制"""
        self.screen.listen()
        self.screen.onkey(lambda: self.set_direction("up"), "Up")
        self.screen.onkey(lambda: self.set_direction("down"), "Down")
        self.screen.onkey(lambda: self.set_direction("left"), "Left")
        self.screen.onkey(lambda: self.set_direction("right"), "Right")
        self.screen.onkey(self.restart_game, "r")  # 按 R 重新开始
        self.screen.onkey(self.toggle_pause, "space")  # 按空格暂停/继续
    
    def set_direction(self, direction):
        """设置蛇的移动方向（防止直接反向移动）"""
        if (direction == "up" and self.direction != "down") or \
           (direction == "down" and self.direction != "up") or \
           (direction == "left" and self.direction != "right") or \
           (direction == "right" and self.direction != "left"):
            self.next_direction = direction
    
    def move(self):
        """移动蛇"""
        # 更新方向
        self.direction = self.next_direction
        
        # 移动蛇身（从尾部开始，每个部分移动到前一个部分的位置）
        for i in range(len(self.snake)-1, 0, -1):
            x = self.snake[i-1].xcor()
            y = self.snake[i-1].ycor()
            self.snake[i].goto(x, y)
        
        # 移动蛇头
        if self.direction == "up":
            self.head.sety(self.head.ycor() + GRID_SIZE)
        elif self.direction == "down":
            self.head.sety(self.head.ycor() - GRID_SIZE)
        elif self.direction == "left":
            self.head.setx(self.head.xcor() - GRID_SIZE)
        elif self.direction == "right":
            self.head.setx(self.head.xcor() + GRID_SIZE)
    
    def spawn_food(self):
        """在随机位置生成食物"""
        # 计算可能的坐标（网格对齐）
        x = random.randint(-WIDTH//2 + GRID_SIZE, WIDTH//2 - GRID_SIZE)
        y = random.randint(-HEIGHT//2 + GRID_SIZE, HEIGHT//2 - GRID_SIZE)
        x = (x // GRID_SIZE) * GRID_SIZE
        y = (y // GRID_SIZE) * GRID_SIZE
        
        # 确保食物不在蛇身上
        for segment in self.snake:
            if segment.distance(x, y) < GRID_SIZE/2:
                return self.spawn_food()  # 递归直到找到合适位置
        
        self.food.goto(x, y)
    
    def check_collision(self):
        """检查碰撞"""
        # 检查是否撞墙
        x, y = self.head.pos()
        if (x < -WIDTH//2 or x > WIDTH//2 - GRID_SIZE or 
            y < -HEIGHT//2 or y > HEIGHT//2 - GRID_SIZE):
            return True
        
        # 检查是否撞到自己
        for segment in self.snake[1:]:
            if self.head.distance(segment) < GRID_SIZE/2:
                return True
        
        return False
    
    def check_food_collision(self):
        """检查是否吃到食物"""
        if self.head.distance(self.food) < GRID_SIZE/2:
            # 增加分数
            self.score += 10
            self.update_score()
            
            # 添加新的蛇身段
            self.add_segment()
            
            # 生成新食物
            self.spawn_food()
            return True
        return False
    
    def add_segment(self):
        """添加新的蛇身段"""
        segment = turtle.Turtle()
        segment.shape("square")
        segment.color("green")
        segment.penup()
        segment.speed(0)
        
        # 新段落在蛇尾位置
        tail = self.snake[-1]
        segment.goto(tail.xcor(), tail.ycor())
        self.snake.append(segment)
    
    def update_score(self):
        """更新分数显示"""
        self.score_display.clear()
        self.score_display.write(f"分数: {self.score}", align="center", font=("Arial", 16, "bold"))
    
    def show_game_over(self):
        """显示游戏结束画面"""
        game_over = turtle.Turtle()
        game_over.speed(0)
        game_over.color("white")
        game_over.penup()
        game_over.hideturtle()
        game_over.goto(0, 0)
        game_over.write("游戏结束!", align="center", font=("Arial", 24, "bold"))
        
        restart_text = turtle.Turtle()
        restart_text.speed(0)
        restart_text.color("yellow")
        restart_text.penup()
        restart_text.hideturtle()
        restart_text.goto(0, -50)
        restart_text.write("按 R 重新开始，按空格键退出", align="center", font=("Arial", 14, "normal"))
    
    def restart_game(self):
        """重新开始游戏"""
        if self.game_over:
            # 清除屏幕
            self.screen.clear()
            
            # 重新初始化游戏
            self.__init__()
            self.run()
    
    def toggle_pause(self):
        """暂停/继续游戏（简单实现）"""
        if not self.game_over:
            # 这里可以添加暂停逻辑
            pass
    
    def run(self):
        """运行游戏主循环"""
        self.game_over = False
        
        while not self.game_over:
            self.screen.update()
            time.sleep(DELAY)
            
            self.move()
            
            # 检查碰撞
            if self.check_collision():
                self.game_over = True
                self.show_game_over()
                break
            
            # 检查食物
            self.check_food_collision()
        
        # 游戏结束后保持窗口打开
        self.screen.mainloop()

def main():
    """主函数"""
    print("=" * 40)
    print("贪吃蛇游戏")
    print("=" * 40)
    print("控制说明:")
    print("↑ 上箭头: 向上移动")
    print("↓ 下箭头: 向下移动")
    print("← 左箭头: 向左移动")
    print("→ 右箭头: 向右移动")
    print("R: 重新开始游戏")
    print("空格键: 暂停/继续")
    print("=" * 40)
    print("游戏即将开始...")
    
    game = SnakeGame()
    game.run()

if __name__ == "__main__":
    main()