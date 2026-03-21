def remove_Occ(string, char):
  result = ""
  for i in range(len(string)):
    if string[i] != char:
      result += string[i]
  return result

assert remove_Occ("hello","l") == "heo"
assert remove_Occ("abcda","a") == "bcd"
assert remove_Occ("PHP","P") == "H"
