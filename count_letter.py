sentence = input("your sentence:").lower()
letter_list = sentence.replace(" ", "")
alphabet = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
            "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z")
letter_compare = {}

for word in alphabet:
    for key in letter_list:
        if word == key:
            if word in letter_compare:
                letter_compare[word] += 1
            else:
                letter_compare[word] = 1

for k,v in letter_compare.items():
    print("{0}: {1}". format(k,v))











