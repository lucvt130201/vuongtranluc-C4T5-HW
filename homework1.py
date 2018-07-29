inventory = {
    "gold": 500,
    "pouch": ["flint", "twine", "gemstone"],
    "backpack": ['xylophone', 'dagger', 'bedroll', 'bread loaf'],
}

for key in inventory:
    print(key, inventory[key], sep = ': ')
print()

inventory["pocket"] = ['seashell','strange barry', 'lint']
for key in inventory:
    print(key, inventory[key], sep = ': ')
print()

a = inventory["backpack"].remove('dagger')
for key in inventory:
    print(key, inventory[key], sep = ': ')
print()


inventory['gold'] += 50
for key in inventory:
    print(key, inventory[key], sep = ': ')
print()


