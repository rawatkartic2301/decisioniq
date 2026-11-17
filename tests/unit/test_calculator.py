"""Unit tests for safe arithmetic calculator tool."""

import pytest

from decisioniq.tools.calculator_tool import safe_calculate


def test_safe_arithmetic_operations():
    assert safe_calculate("2 + 2") == 4
    assert safe_calculate("10 - 4.5") == 5.5
    assert safe_calculate("3 * 5") == 15
    assert safe_calculate("20 / 4") == 5.0
    assert safe_calculate("(10 + 5) * 2") == 30
    assert safe_calculate("2 ** 3") == 8
    assert safe_calculate("1,500 + 2,500") == 4000


def test_calculator_rejects_arbitrary_code():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('ls')")

    with pytest.raises(ValueError):
        safe_calculate("open('file.txt').read()")

    with pytest.raises(ValueError):
        safe_calculate("x = 5")
