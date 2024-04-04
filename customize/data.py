from typeguard import typechecked
from typing import int
@typechecked
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers(5, 10)  # No type error, returns 15
result = add_numbers(5, "10")  # Type error, raises TypeCheckerError