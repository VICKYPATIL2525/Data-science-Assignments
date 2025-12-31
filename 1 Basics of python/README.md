# Assignment 1: Basics of Python

## Topics Covered

### 1. Functions
- Function definition with `def`
- Parameters and return values
- Docstrings

### 2. Loops
- `for` loop with `range()`
- Loop with conditions

### 3. Conditionals
- `if`, `else` statements
- Multiple conditions

### 4. Data Structures
- **Dictionary**: `{}`, key-value pairs
- **List**: `[]`

### 5. String Operations
- `split()` - split string into words
- `lower()` - convert to lowercase
- String slicing `[::-1]` - reverse string

### 6. Modules Used
- `random` - `randint()` for random numbers
- `re` - `sub()` for regex pattern replacement

### 7. Operators
- `%` - modulo (remainder)
- `**` - power/exponent
- `in` - membership check

### 8. Input/Output
- `input()` - get user input
- `print()` - display output
- Type conversion with `int()`

---

## Interview Questions

### Q1: What is the difference between list and tuple?
**List:**
- Mutable (can change)
- `[1, 2, 3]`
- Slower

**Tuple:**
- Immutable (cannot change)
- `(1, 2, 3)`
- Faster

### Q2: Explain `range()` function
```python
range(5)        # 0, 1, 2, 3, 4
range(2, 5)     # 2, 3, 4
range(0, 10, 2) # 0, 2, 4, 6, 8
```

### Q3: How to reverse a string?
```python
s = "hello"
reversed_s = s[::-1]  # "olleh"
```

### Q4: What is `if __name__ == "__main__"`?
- Code runs only when file is executed directly
- Not when imported as module

### Q5: Dictionary methods?
```python
d = {'a': 1, 'b': 2}
d.keys()        # dict_keys(['a', 'b'])
d.values()      # dict_values([1, 2])
d.items()       # dict_items([('a', 1), ('b', 2)])
d.get('a')      # 1
d.get('c', 0)   # 0 (default value)
```

### Q6: What is the difference between `append()` and `extend()`?
```python
list1 = [1, 2, 3]
list1.append([4, 5])   # [1, 2, 3, [4, 5]]

list2 = [1, 2, 3]
list2.extend([4, 5])   # [1, 2, 3, 4, 5]
```

### Q7: How does `split()` work?
```python
text = "hello world test"
text.split()           # ['hello', 'world', 'test']
text.split('o')        # ['hell', ' w', 'rld test']
```

### Q8: What is modulo operator `%`?
```python
10 % 3   # 1 (remainder)
10 % 2   # 0 (even number)
11 % 2   # 1 (odd number)
```

### Q9: Difference between `==` and `is`?
```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

a == b   # True (same values)
a is b   # False (different objects)
a is c   # True (same object)
```

### Q10: What are mutable and immutable types?
**Mutable:** list, dict, set
**Immutable:** int, float, str, tuple

```python
# Immutable
s = "hello"
s[0] = "H"  # Error!

# Mutable
lst = [1, 2, 3]
lst[0] = 10  # Works!
```

### Q11: How to check if number is prime?
```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
```

### Q12: What is `random.randint()`?
```python
import random
random.randint(1, 10)  # Random number between 1 and 10 (inclusive)
```

### Q13: What is regex `re.sub()`?
```python
import re
text = "Hello! World?"
cleaned = re.sub(r'[^a-zA-Z0-9]', '', text)  # "HelloWorld"
# Removes all non-alphanumeric characters
```

### Q14: How to count occurrences in a string/list?
```python
# Method 1: Dictionary
word_count = {}
for word in words:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

# Method 2: get()
word_count = {}
for word in words:
    word_count[word] = word_count.get(word, 0) + 1

# Method 3: Counter
from collections import Counter
word_count = Counter(words)
```

### Q15: Difference between `/` and `//`?
```python
7 / 2    # 3.5 (float division)
7 // 2   # 3 (integer division)
```

### Q16: How to iterate through dictionary?
```python
d = {'a': 1, 'b': 2, 'c': 3}

# Keys
for key in d:
    print(key)

# Values
for value in d.values():
    print(value)

# Key-value pairs
for key, value in d.items():
    print(key, value)
```

### Q17: String formatting methods?
```python
name = "John"
age = 25

# f-string (Python 3.6+)
print(f"{name} is {age} years old")

# format()
print("{} is {} years old".format(name, age))

# % operator
print("%s is %d years old" % (name, age))
```

### Q18: What is type conversion?
```python
int("123")      # 123
str(123)        # "123"
float("12.5")   # 12.5
list("abc")     # ['a', 'b', 'c']
```

### Q19: How to handle multiple conditions?
```python
# AND
if x > 0 and x < 10:
    print("x is between 0 and 10")

# OR
if x < 0 or x > 10:
    print("x is outside 0-10")

# NOT
if not x:
    print("x is False/None/0/empty")
```

### Q20: What is the `in` operator?
```python
# String
"hello" in "hello world"  # True

# List
2 in [1, 2, 3]  # True

# Dictionary (checks keys)
'a' in {'a': 1, 'b': 2}  # True
```

---

## Files
- `basicpython.py` - Solutions to all 5 exercises
- `Basics of python.docx` - Assignment questions
