def count_characters(string):
    
  char_count = {}
  #  kiểm tra xem ký tự hiện tại (char) có đã tồn tại trong từ điển char_count dưới dạng key hay chưa
  for char in string:
    if char in char_count:
      char_count[char] += 1
    else:
      char_count[char] = 1
      
  return char_count

string = "Happiness"
print(count_characters(string))

string = "smiles"
print(count_characters(string))

