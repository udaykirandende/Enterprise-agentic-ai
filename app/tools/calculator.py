import ast
import math
import operator
import re


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _normalize_expression(expression: str) -> str:
    """Convert a limited set of natural-language operators to symbols."""
    result = expression.strip().lower()

    replacements = [
        (r"\bdivided\s+by\b", "/"),
        (r"\bmultiplied\s+by\b", "*"),
        (r"\btimes\b", "*"),
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\bmodulo\b", "%"),
        (r"\bmod\b", "%"),
    ]

    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)

    # Accept a leading instruction without interpreting arbitrary language.
    result = re.sub(r"^(calculate|compute|what is)\s+", "", result).strip()
    result = result.rstrip(" ?")

    return result


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numbers are allowed.")
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large.")

        result = _BINOPS[type(node.op)](left, right)

        if not math.isfinite(result) or abs(result) > 1e100:
            raise ValueError("Result outside allowed range.")

        return result

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_evaluate(node.operand))

    raise ValueError("Unsupported arithmetic expression.")


def calculate(expression: str) -> int | float:
    if not expression or len(expression) > 200:
        raise ValueError("Expression must contain 1–200 characters.")

    normalized = _normalize_expression(expression)

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid arithmetic expression.") from exc

    return _evaluate(tree)
