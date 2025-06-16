
def fibonacci(n):
    """Return the nth Fibonacci number (0-indexed)."""
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative integers.")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def factorial(n):
    """Return the factorial of n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative integers.")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def my_pow(base, exp):
    """Compute base raised to the power exp."""
    if exp == 0:
        return 1
    result = 1
    is_negative = exp < 0
    exp = abs(exp)
    while exp > 0:
        if exp % 2 == 1:
            result *= base
        base *= base
        exp //= 2
    return 1 / result if is_negative else result

def quick_sort(arr):
    """Sort a list using the Quick Sort algorithm."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)