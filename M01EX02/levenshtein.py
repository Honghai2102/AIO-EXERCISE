def del_cost(char):
    return 1


def ins_cost(char):
    return 1


def sub_cost(char1, char2):
    if char1 == char2:
        return 0
    return 1


def leven_dist_input():
    target_string = input(f"Enter target string: ")
    source_string = input(f"Enter sooure string: ")

    target_string = "#" + target_string
    source_string = "#" + source_string
    
    return target_string, source_string


def leven_dist_function():
    target_string, source_string = leven_dist_input()
    len_target = len(target_string)
    len_source = len(source_string)
    d_matrix = [[None] * len_target for _ in range(len_source)]

    for _ in range(len_target):
        d_matrix[0][_] = _
    
    for _ in range(1, len_source):
        d_matrix[_][0] = _

    for r in range(1, len_source):
        for c in range(1, len_target):
            d_del = d_matrix[r - 1][c] + del_cost(source_string[r])
            d_ins = d_matrix[r][c - 1] + ins_cost(target_string[c])
            d_sub = d_matrix[r - 1][c - 1] + sub_cost(source_string[r], target_string[c])
            d_matrix[r][c] = min(d_del, d_ins, d_sub)
            
    leven_dist = d_matrix[len_source - 1][len_target - 1]
    print(f"Levenshtein distance between {target_string[1:]} and {source_string[1:]} is {leven_dist}")

    return True


if __name__ == "__main__":
    leven_dist_function()           # Input: target string and source string
                                    # Output: levenshtein distance
