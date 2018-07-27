height = float(input("your height(cm)= "))

h = height/100

weight = float(input("your weight(kg)"))

BMI = weight / (h * h)

print("according to your data, doctor said that:")

if BMI <= 16:
    print("you are severely underweight")
elif BMI <= 18.5:
    print("you are underweight")
elif BMI <= 25:
    print("you are normal")
elif BMI <= 30:
    Print("you are overweight")
else:
    print("you are obese")