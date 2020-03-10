import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

resize = T.Compose([T.ToPILImage(),
                    T.Resize(300, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(food, enemy, player, d, SIZE= 10, show=False):
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
    env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
    env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
    env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
    img = env.transpose((2, 0, 1))
    if (show):
        cv2.imshow("image", env)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255
    screen = torch.from_numpy(img)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)