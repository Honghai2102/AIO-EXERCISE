import random
import math


def check_loss_name(loss_name):
    loss_name = loss_name.lower()
    loss_name = loss_name.replace(" ", "")
    if loss_name == "mae" or loss_name == "mse" or loss_name == "rmse":
        return True
    return False


def Cal_Loss_Regression(n, loss_name, y, y_hat):
    total_loss = 0.0
    for _ in range(n):
        if loss_name == "mae":
            loss = abs(y[_] - y_hat[_])
        else:
            loss = (y[_] - y_hat[_])**2
        total_loss += loss
        print(f"loss name: {loss_name}, sample: {_}, pred: {y_hat[_]}, target: {y[_]}, loss: {loss}")
    if loss_name == "rmse":
        return math.sqrt(total_loss/n)
    return total_loss/n


if __name__ == "__main__":
    print("Please enter value x and activation function")
    while True:
        while True:
            n = input("Input number of samples (integer number) which are generated: ")
            if n.isnumeric():
                n = int(n)
                break
            else:
                print("number of samples must be an integer number")
                print("Please enter number of samples again")
        while True:
            loss_name = input("Input loss name (MAE, MSE, RMSE): ")
            check_loss_res = check_loss_name(loss_name)
            if check_loss_res == True:
                loss_name = loss_name.lower()
                loss_name = loss_name.replace(" ", "")
                break
            else:
                print(f"{loss_name} is not supported")
                print("Please enter loss regression name again")
        break

    y = []
    y_hat = []
    for _ in range(n):
        y.append(random.uniform(0, 10))
        y_hat.append(random.uniform(0, 10))
    print(f"final {loss_name}: {Cal_Loss_Regression(n, loss_name, y, y_hat)}")
