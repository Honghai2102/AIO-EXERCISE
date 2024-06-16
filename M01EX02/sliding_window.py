def sliding_window(num_list, k):

    max_values = []
    left = 0
    right = k - 1
    # tìm số lớn nhất trong sliding window hiên tai
    for _ in num_list:
        if right < len(num_list):
            max_value = max(num_list[left:right + 1])
            max_values.append(max_value)
     # di chuyển cửa sổ trươt sang phải
        left += 1
        right += 1

    return max_values


num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
k = 3
print(sliding_window(num_list, k))
