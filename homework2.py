price = {
    "banana": 4,
    "apple": 2,
    "orange": 1.5,
    "pear": 3

}

stock = {
    "banana": 6,
    "apple": 0,
    "orange": 32,
    "pear": 15

}

for i, item in zip(price, stock):
    print(i)
    print("price:", price[i])
    print("stock",stock[item])
    print()

total = 0

for i, item  in zip(price, stock):
    a = price[i] * stock[i]
    total += a
print(total)