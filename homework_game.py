import pygame

pygame.init()
pygame.display.set_caption("ponggame version 2")
SIZE = (600, 600)
canvas = pygame.display.set_mode(SIZE)
BG_COLOR = (70,95,70)
clock = pygame.time.Clock()

paddle_image = pygame.image.load("assets/paddle.png")
ball_image = pygame.image.load("assets/ball.png")

x1 = 0
y1 = 250
x2 = 570
y2 = 250
ball_x = 285
ball_y = 300
w_pressed = False
s_pressed = False
up_pressed = False
down_pressed = False
ball_x_v = 1.5
ball_y_v = 2


loop = True

while loop:
    events = pygame.event.get()
    for e in events:
        if e.type == pygame.QUIT:
            loop = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_w:
                w_pressed = True
            elif e.key == pygame.K_s:
                s_pressed = True
            elif e.key == pygame.K_UP:
                up_pressed = True
            elif e.key == pygame.K_DOWN:
                down_pressed = True

        elif e.type == pygame.KEYUP:
            if  e.key == pygame.K_w:
                w_pressed = False
            elif e.key == pygame.K_s:
                s_pressed = False
            elif e.key == pygame.K_DOWN:
                down_pressed = False
            elif e.key == pygame.K_UP:
                up_pressed = False
    ball_x += ball_x_v
    ball_y += ball_y_v

    if y1 <= ball_y <= (y1 + 120) and ball_x <=  (x1 + 30):
        ball_x_v = -ball_x_v
    if y2 <= ball_y <= (y2 +120) and ball_x >= (x2 - 20):
        ball_x_v = - ball_x_v
    if ball_x <= 0 or ball_x >=580:
        ball_x_v = -ball_x_v
    if ball_y <= 0 or ball_y >= 580:
        ball_y_v = -ball_y_v
    if y1 <= 0:
        w_pressed = False
    if y2 <= 0:
        up_pressed = False
    if y1 >= 480:
        s_pressed = False
    if y2 >= 480:
        down_pressed = False






    if w_pressed:
        y1 += -5
    elif s_pressed:
        y1 += 5
    elif up_pressed:
        y2 += -5
    elif down_pressed:
        y2 += 5

    canvas.fill(BG_COLOR)
    canvas.blit(paddle_image, (x1, y1))
    canvas.blit(paddle_image, (x2, y2))
    canvas.blit(ball_image, (ball_x, ball_y))
    clock.tick(60)

    pygame.display.flip()