"""Safe AST-based arithmetic calculator tool without eval() vulnerabilities."""

import ast
import operator
from typing import Union

# Allowed operators
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression: str) -> Union[int, float]:
    """Safely evaluate arithmetic expressions using an Abstract Syntax Tree (AST)."""
    cleaned = expression.strip().replace(",", "")
    if not cleaned:
        raise ValueError("Empty calculation expression.")

    try:
        node = ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid arithmetic expression: {e}") from e

    def _eval(node_item):
        if isinstance(node_item, ast.Expression):
            return _eval(node_item.body)
        elif isinstance(node_item, ast.Constant):  # Python 3.8+ numbers
            if isinstance(node_item.value, (int, float)):
                return node_item.value
            raise ValueError(f"Unsupported constant type: {type(node_item.value)}")
        elif isinstance(node_item, ast.BinOp):
            left = _eval(node_item.left)
            right = _eval(node_item.right)
            op_type = type(node_item.op)
            if op_type in OPERATORS:
                return OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type}")
        elif isinstance(node_item, ast.UnaryOp):
            operand = _eval(node_item.operand)
            op_type = type(node_item.op)
            if op_type in OPERATORS:
                return OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        else:
            raise ValueError(f"Unsupported AST node expression: {type(node_item).__name__}")

    return _eval(node)
