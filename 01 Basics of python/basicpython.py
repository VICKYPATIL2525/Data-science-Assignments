"""
Python Programming Assignment - Basics of Python
Simple and straightforward solutions to 5 exercises
"""

import random
import re

# Exercise 1: Prime Number Checker
def is_prime(n):
    """Check if a number is prime"""
    if n <= 1:
        return False

    for i in range(2, n):
        if n % i == 0:
            return False
    return True


# Exercise 2: Product of Random Numbers Quiz
def product_quiz():
    """Quiz user on product of two random numbers"""
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)

    print(f"What is {num1} × {num2}?")
    answer = int(input("Your answer: "))

    if answer == num1 * num2:
        print("Correct!")
    else:
        print(f"Wrong! The answer is {num1 * num2}")


# Exercise 3: Squares of Even Numbers (100-200)
def print_squares():
    """Print squares of even numbers from 100 to 200"""
    print("Squares of even numbers from 100 to 200:")
    for num in range(100, 201):
        if num % 2 == 0:
            print(f"{num}² = {num**2}")


# Exercise 4: Word Counter
def count_words(text):
    """Count frequency of each word in text"""
    words = text.split()
    word_count = {}

    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    return word_count


# Exercise 5: Palindrome Checker
def is_palindrome(text):
    """Check if a string is a palindrome (ignoring spaces, punctuation, case)"""
    # Clean the text: remove non-alphanumeric and make lowercase
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text.lower())
    # Check if it reads the same forwards and backwards
    return cleaned == cleaned[::-1]


# Main function to test all exercises
def main():
    # Test Exercise 1
    print("=== Exercise 1: Prime Numbers ===")
    test_nums = [2, 3, 4, 17, 20, 29]
    for num in test_nums:
        print(f"{num} is prime: {is_prime(num)}")
    print()

    # Test Exercise 2
    print("=== Exercise 2: Product Quiz ===")
    # product_quiz()  # Uncomment to run the interactive quiz
    print("(Interactive quiz - uncomment to test)")
    print()

    # Test Exercise 3
    print("=== Exercise 3: Squares of Even Numbers ===")
    print_squares()
    print()

    # Test Exercise 4
    print("=== Exercise 4: Word Counter ===")
    text = "This is a sample text. This text will be used to demonstrate the word counter."
    result = count_words(text)
    for word, count in result.items():
        print(f"'{word}': {count}")
    print()

    # Test Exercise 5
    print("=== Exercise 5: Palindrome Checker ===")
    test_strings = ["racecar", "hello", "A man a plan a canal Panama"]
    for s in test_strings:
        print(f"'{s}' is palindrome: {is_palindrome(s)}")


if __name__ == "__main__":
    main()
