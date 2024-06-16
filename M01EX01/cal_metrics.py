
def cal_metrics(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, f1_score


def check_metrics(tp, fp, fn):
    if not tp.isnumeric():
        return print("tp must be int")
    if not fp.isnumeric():
        return print("fp must be int")
    if not fn.isnumeric():
        return print("fn must be int")
    if not int(tp) > 0 or not int(fp) > 0 or not int(fn) > 0:
        return print("tp and fp and fn must be greater than zero")
    return True


if __name__ == "__main__":
    print("Please enter tp, fp and fn")
    while True:
        tp = input("tp = ")
        fp = input("fp = ")
        fn = input("fn = ")
        if check_metrics(tp, fp, fn) == True:
            tp = int(tp)
            fp = int(fp)
            fn = int(fn)
            break
        else:
            print("Please enter tp, fp and fn again")

    precision, recall, f1_score = cal_metrics(tp, fp, fn)
    print(f"precision is {round(precision, 3)}\nrecall is {round(recall, 3)}\nf1_score is {round(f1_score, 3)}")
    