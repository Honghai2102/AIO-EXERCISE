def md_nre(y, y_hat, n, p):
    return ((y ** (1 / n)) - (y_hat ** (1 / n))) ** p


def check_input(y, y_hat, n, p):
    try:
        float(y)
        float(y_hat)
        float(n)
        float(p)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    print("Please enter value y, y_hat, n and p")
    while True:
        y = input("y = ")
        y_hat = input("y_hat = ")
        n = input("n = ")
        p = input("p = ")
        if check_input(y, y_hat, n, p) == True:
            y = float(y)
            y_hat = float(y_hat)
            n = float(n)
            p = float(p)
            break
        else:
            print("Value y, y_hat, n and p must be numbers")
            print("Please enter value y, y_hat, n and p again")

    print(f"md_nre_single_sample (y={y}, y_hat={y_hat}, n={n}, p={p}): {md_nre(y, y_hat, n, p)}")