# countdown_eval.py
import ast
import re
from collections import Counter
from fractions import Fraction
from typing import Optional, Tuple, List


_ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_answer_block(response: str) -> Optional[str]:
    """Return the full content inside <answer>...</answer>. Must exist."""
    m = _ANSWER_BLOCK_RE.search(response)
    if not m:
        return None
    return m.group(1).strip()


def split_equation(answer_block: str) -> Tuple[str, Optional[str]]:
    """
    Split '<answer> expr = rhs </answer>' into (expr, rhs).
    If '=' not found, treat whole as expr.
    """
    text = answer_block.strip()
    if "=" not in text:
        return text, None
    left, right = text.rsplit("=", 1)
    return left.strip(), right.strip()


# ---------- Safe exact evaluation (Fraction) ----------

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _eval_fraction(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_fraction(node.body)

    # numbers
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        # we still accept float literals but turn them exact via Fraction(str(x))
        # (ideally they won't appear)
        if isinstance(node.value, int):
            return Fraction(node.value, 1)
        return Fraction(str(node.value))
    if isinstance(node, ast.Num):  # py<3.8
        if isinstance(node.n, int):
            return Fraction(node.n, 1)
        return Fraction(str(node.n))

    # unary +/- number/expression
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        val = _eval_fraction(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val

    # binary ops
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        a = _eval_fraction(node.left)
        b = _eval_fraction(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            if b == 0:
                raise ZeroDivisionError("division by zero")
            return a / b

    raise ValueError(f"Disallowed expression: {ast.dump(node, include_attributes=False)}")


def safe_eval_to_fraction(expr: str) -> Fraction:
    """
    Safely evaluate arithmetic expression with + - * / and parentheses.
    Returns exact Fraction.
    """
    tree = ast.parse(expr, mode="eval")
    return _eval_fraction(tree)


# ---------- Extract integer literals used in expression ----------

def _extract_int_literals(node: ast.AST) -> List[int]:
    """
    Extract integer literals appearing in the expression AST.
    Supports unary negative integers like -3.
    """
    ints: List[int] = []

    def visit(n: ast.AST):
        # literal int
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            ints.append(int(n.value))
            return
        if isinstance(n, ast.Num) and isinstance(n.n, int):  # py<3.8
            ints.append(int(n.n))
            return

        # unary -int
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            if isinstance(n.operand, ast.Constant) and isinstance(n.operand.value, int):
                ints.append(-int(n.operand.value))
                return
            if isinstance(n.operand, ast.Num) and isinstance(n.operand.n, int):
                ints.append(-int(n.operand.n))
                return

        # recurse
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    return ints


def extract_used_numbers(expr: str) -> List[int]:
    tree = ast.parse(expr, mode="eval")
    return _extract_int_literals(tree)


# ---------- Main verifier ----------

def verify_countdown_response(response: str, target: int, nums: list[int]) -> bool:
    """
    Strict verification:
    - MUST contain <answer>...</answer>
    - Expression must use each number in nums exactly once (multiset equality)
    - Exact evaluation of expression equals target (as integer)
    """
    answer_block = extract_answer_block(response)
    if answer_block is None:
        return False

    expr, rhs = split_equation(answer_block)
    if not expr:
        return False

    # Check used numbers exactly match nums (multiset)
    try:
        used_numbers = extract_used_numbers(expr)
    except Exception:
        return False

    if Counter(used_numbers) != Counter(nums):
        return False

    # Evaluate expression exactly
    try:
        val = safe_eval_to_fraction(expr)
    except Exception:
        return False

    if val != Fraction(target, 1):
        return False

    # Optional: if RHS exists and is a number, also require it equals target (helps catch weird formatting)
    if rhs is not None and rhs != "":
        m = re.search(r"[-+]?\d+$", rhs)
        if m:
            if int(m.group(0)) != target:
                return False

    return True