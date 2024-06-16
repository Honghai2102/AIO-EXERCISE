import requests


def file_txt_input():
    url = "https://drive.google.com/uc?id=1IBScGdW2xlNsc9v5zSAya548kNgiOrko"
    response = requests.get(url)
    word_count_list = response.text
    word_count_list = word_count_list.lower().split()
    return word_count_list


def word_count_function():
    word_count_list = file_txt_input()
    keys_list = list(set(word_count_list))
    values_list = []

    for keys_list_item in keys_list:
        value_count = 0
        for word_count_list_item in word_count_list:
            if word_count_list_item == keys_list_item:
                value_count += 1
        values_list.append(value_count)
        
    word_count_dict = dict(zip(keys_list, values_list))
    word_count_dict = dict(sorted(word_count_dict.items(), key=lambda item: item[1]))
    print(f"Count each word in file.txt: {word_count_dict}")

    return True


if __name__ == "__main__":
    word_count_function()           # Input: a file.txt
                                    # Output: a dictionary (value ascending)







