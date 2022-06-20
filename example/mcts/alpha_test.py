
import string
from itertools import takewhile
  
# generating alphabets in random order
li = list(string.ascii_lowercase)
  
print("The original list list is :")
print(li)
  
# consider the element until
# 'e' or 'i' or 'o' is encountered
new_li = list(takewhile(lambda x:x <= 'c',
                        li))
  
print("\nThe new list is :")
print(new_li)

# import string
# from itertools import takewhile

# goal_list = list(string.ascii_uppercase)
# alphabet = ['C']
# result = list(takewhile(lambda x: x in alphabet, goal_list))
# print(result, goal_list)