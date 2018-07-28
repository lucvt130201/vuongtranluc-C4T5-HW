shop_item = ["T-Shirt", "Sweeter"]
shop = "wellcome to our shop. what do you want(C, R, U, D)?"
print(shop)

custumer = input("I choose:")

if custumer == "R" or custumer == "r":
    print("Our item is:", *shop_item, sep="," )
elif custumer == "C" or custumer == "c":
    new_item = "Jeans"
    shop_item.append(new_item)
    print("Our item is:", *shop_item, sep=",")
elif custumer == "U" or custumer == "u":
    position = int(input("update position:"))
    new_item = "Skirt"
    shop_item[position] = new_item
    print("Our item is:", *shop_item, sep=",")
elif custumer == "D" or custumer == "d":
    position_del = int(input("delete position:"))
    shop_item.pop(position_del)
    print("Our item is:", *shop_item, sep=",")
else:
    print("Sorry!Your choice is not in our list. Please, try again!!!")




