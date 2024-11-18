with open("secret.daz","r",encoding="utf-8") as file:
    string=file.read()
re_strings=string.split("X")
true_strings = ''.join(chr(int(re_string, 16)) for re_string in re_strings if re_string)
with open("interpretation.txt","w",encoding="utf-8") as file:
    file.write(true_strings)
import re
with open("interpretation.txt","r",encoding="utf-8") as file:
    pure_string=re.sub(r'\s+', '',file.read())
number=len(pure_string)
with open("interpretation.txt","a",encoding="utf-8") as file:
    file.write(f"\n4月5日战伤统计\n<解密人>2024090905001<情报总字数>{number}")