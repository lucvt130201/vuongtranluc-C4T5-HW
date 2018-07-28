from turtle import*

shape("turtle")
speed(0)
colors = ['red', 'blue', 'brown', 'yellow', 'grey']

for item in colors:
    pencolor(item)
    fillcolor(item)
    begin_fill()

    for i in range(2):
        forward(50)
        left(90)
        forward(150)
        left(90)
    forward(50)
    end_fill()



mainloop()