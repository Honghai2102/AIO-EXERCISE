import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    if x <= 0:
        return 0
    return x


def elu(x):
    alpha = 0.01
    if x <= 0:
        return alpha * (math.exp(x) - 1)
    return x


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def check_act_funct(acti_funct_name):
    acti_funct_name = acti_funct_name.lower()
    acti_funct_name = acti_funct_name.replace(" ", "")
    if acti_funct_name == "sigmoid" or acti_funct_name == "relu" or acti_funct_name == "elu":
        return True
    return False


if __name__ == "__main__":
    print("Please enter value x and activation function")
    while True:
        while True:
            x = input("Input x = ")
            if is_number(x) == True:
                x = float(x)
                break
            else:
                print("x must be a number")
                print("Please enter value x again")
        while True:
            acti_funct_name = input("Input activation Function (sigmoid|relu|elu): ")
            check_acti_funct_res = check_act_funct(acti_funct_name)
            if check_acti_funct_res == True:
                acti_funct_name = acti_funct_name.lower()
                acti_funct_name = acti_funct_name.replace(" ", "")
                break
            else:
                print(f"{acti_funct_name} is not supported")
                print("Please enter activation function name again")
        break
    
    if acti_funct_name == "sigmoid":
        print(f"{acti_funct_name}: f({x}) = {sigmoid(x)}")
    elif acti_funct_name == "relu":
        print(f"{acti_funct_name}: f({x}) = {relu(x)}")
    else:    
        print(f"{acti_funct_name}: f({x}) = {elu(x)}")
