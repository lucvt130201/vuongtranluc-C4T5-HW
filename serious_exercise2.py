# 2.1
size = [5,7,300,90,24,50,75]
print("Hello, my name is Luc, and these are my ships' sizes" )
print(size)

# 2.2
shear = max(size)
print("Now my biggest sheep has size", shear, "let's shear it")

# 2.3
size[2] = 8
print("After shearing, hear is my flock")
print(size)

# 2.4
increase = 50
for i in range(7):
    size[i] = size[i] + 50
print("One month has passed, now here is my flock")
print(size)






