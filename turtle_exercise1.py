from turtle import*

shape("turtle")
speed(0)
colors = ['red', 'blue', 'brown', 'yellow', 'grey']
n_angles = 3
for i in colors:
    color(i)
    for i in range(n_angles):
        forward(100)
        left(360/n_angles)


    n_angles +=1
mainloop()

