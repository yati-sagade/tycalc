"""Interpreter for TyCalc expressions.

This module can be used programmatically via the Interp class, but can also be
run directly:

    $ python interp.py -i INPUT_FILE

To start a REPL,

    $ python interp.py

"""

import sys
import argparse
from parsing import Scanner, Parser, ExprVisitor, BinaryExpr, UnaryExpr, Literal, IdentExpr, Expr, Bool
import typechecking
from typechecking import TypeChecker, get_type, is_err


class Val:
    """Base class of all value types."""
    pass


class IntVal(Val):
    """An integer value."""

    def __init__(self, val: int):
        self.val = val

    def __repr__(self):
        return repr(self.val)


class FloatVal(Val):
    """An floating point number value."""

    def __init__(self, val: float):
        self.val = val

    def __repr__(self):
        return repr(self.val)


class BoolVal(Val):
    """A Boolean value."""

    def __init__(self, val: bool):
        self.val = val

    def __repr__(self):
        return 'true' if self.val else 'false'


def make(ty: typechecking.Type, val):
    """Return a `Val` instance of the given type `ty` that wraps `val`"""
    if ty == typechecking.Int():
        return IntVal(val)
    if ty == typechecking.Float():
        return FloatVal(val)
    if ty == typechecking.Bool():
        return BoolVal(val)
    raise Exception(f'Unknown type {ty}')


def _add(l: Val, r: Val, ty: typechecking.Type):
    """Return a `Val` of type `ty` containing the numeric addition of `l` and `r`.

    Assumes `l.val` and `r.val` have numeric types.
    """
    return make(ty, l.val + r.val)


def _sub(l: Val, r: Val, ty: typechecking.Type):
    """Return a `Val` of type `ty` containing the subtraction of `r` from `l`.

    Assumes `l.val` and `r.val` have numeric types.
    """
    return make(ty, l.val - r.val)


def _div(l: Val, r: Val, ty: typechecking.Type):
    """Return a `Val` of type `ty` containing the division of `l` by `r`.

    Assumes `l.val` and `r.val` have numeric types.
    Raises if `r.val` is zero.
    """
    return make(ty, l.val / r.val)


def _mul(l: Val, r: Val, ty: typechecking.Type):
    """Return a `Val` of type `ty` containing the multiplication of `l` by `r`.

    Assumes `l.val` and `r.val` have numeric types.
    """
    return make(ty, l.val * r.val)


def _mod(l: Val, r: Val, ty: typechecking.Type):
    """Return a `Val` of type `ty` containing the remainder of the division `l.val / r.val`.

    Assumes `l.val` and `r.val` are integers.
    Raises if `r.val` is zero.
    """
    return make(ty, l.val % r.val)


def _exp(l: Val, r: Val, ty: typechecking.Type):
    """Return a `Val` of type `ty` containing the result of raising `l.val` to the `r.val`th power

    Assumes `l.val` and `r.val` have numeric types.
    """
    return make(ty, l.val**r.val)


def _logic_and(l: Val, r: Val, ty):
    """Return a `BoolVal` containing the result of `l.val && r.val`.

    Assumes `l.val` and `r.val` have boolean types.
    Requires `ty` to be typechecking.Bool.
    """
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val and r.val)


def _logic_or(l: Val, r: Val, ty: typechecking.Type):
    """Return a `BoolVal` containing the result of `l.val || r.val`.

    Assumes `l.val` and `r.val` have boolean types.
    Requires `ty` to be typechecking.Bool.
    """
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val or r.val)


def _cmp_lt(l: Val, r: Val, ty: typechecking.Type):
    """Return a `BoolVal` containing the result of `l.val < r.val`.

    Assumes `l.val` and `r.val` have numeric types.
    Requires `ty` to be typechecking.Bool.
    """
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val < r.val)


def _cmp_le(l: Val, r: Val, ty: typechecking.Type):
    """Return a `BoolVal` containing the result of `l.val <= r.val`.

    Assumes `l.val` and `r.val` have numeric types.
    Requires `ty` to be typechecking.Bool.
    """
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val <= r.val)


def _cmp_gt(l: Val, r: Val, ty: typechecking.Type):
    """Return a `BoolVal` containing the result of `l.val > r.val`.

    Assumes `l.val` and `r.val` have numeric types.
    Requires `ty` to be typechecking.Bool.
    """
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val > r.val)


def _cmp_ge(l: Val, r: Val, ty: typechecking.Type):
    """Return a `BoolVal` containing the result of `l.val >= r.val`.

    Assumes `l.val` and `r.val` have numeric types.
    Requires `ty` to be typechecking.Bool.
    """
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val >= r.val)


# Op table for binary operations; operator => func(lhs, rhs, output_type)
BINARY_OPS = {
    '+': _add,
    '-': _sub,
    '*': _mul,
    '/': _div,
    '%': _mod,
    '**': _exp,
    '||': _logic_or,
    '&&': _logic_and,
    '<': _cmp_lt,
    '<=': _cmp_le,
    '>': _cmp_gt,
    '>=': _cmp_ge,
}


def _unary_plus(x: Val, ty: typechecking.Type):
    """Return a `Val` representing the unary plus operator applied to `x.val`, i.e., `+x`.

    Assumes `x.val` has a numeric type.
    """
    return make(ty, x.val)


def _unary_minus(x: Val, ty: typechecking.Type):
    """Return a `Val` representing the unary minus operator applied to `x.val`, i.e., `-x`.

    Assumes `x.val` has a numeric type.
    """
    return make(ty, -x.val)


def _logic_not(x: BoolVal, ty: typechecking.Type):
    """Return a `Val` representing the unary not operator applied to `x.val`, i.e., `!x`.

    Assumes `x.val` has a boolean type.
    Requires `ty` to be typechecking.Bool.
    """
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(not x.val)


# Op table for unary operations; operator => func(operand, output_type)
UNARY_OPS = {
    '+': _unary_plus,
    '-': _unary_minus,
    '!': _logic_not,
}


class Interp(ExprVisitor):
    """The TyCalc interpreter."""

    def __init__(self, debug=False):
        """ctor.

        When debug=True, outputs diagnostic messages to stdout as expressions are evaluated.
        """
        self.enable_debug = debug
        self.env = {}

    def evaluate(self, expr: Expr) -> Val:
        """Evaluate an expression and return the result wrapped in a `Val`.

        This is the entry point for evaluation of individual typed `Exprs`. See
        run() for the entry point for evaluation of textual programs.
        """
        return expr.accept(self)

    def visit_binary_expr(self, expr: BinaryExpr):
        """Evaluate a binary expression.
        
        Implements ExprVisitor::visit_binary_expr().
        """
        self.debug('visit_binary_expr')
        self.debug(f'eval {expr}')
        if expr.op.value == '=':
            assert isinstance(expr.left, IdentExpr)
            rhs = self.evaluate(expr.right)
            self.env[expr.left.ident] = rhs
            return rhs
        lhs = self.evaluate(expr.left)
        rhs = self.evaluate(expr.right)
        self.debug(f'  left {expr.left} = {lhs}')
        self.debug(f'  left {expr.right} = {rhs}')
        return BINARY_OPS[expr.op.value](lhs, rhs, get_type(expr))

    def visit_unary_expr(self, expr: UnaryExpr):
        """Evaluate a unary expression.
        
        Implements ExprVisitor::visit_unary_expr().
        """
        self.debug('visit_unary_expr')
        arg = self.evaluate(expr.arg)
        return UNARY_OPS[expr.op.value](arg, get_type(expr))

    def visit_literal(self, expr: Literal):
        """Evaluate a literal expression.
        
        Implements ExprVisitor::visit_literal().
        """
        self.debug('visit_literal')
        return make(get_type(expr), expr.value.value)

    def visit_ident_expr(self, expr: IdentExpr):
        """Evaluate an identifier reference expression.
        
        Note that an identifier reference is different from assignment to an
        identifier, which is a binary expression.

        Implements ExprVisitor::visit_ident_expr().
        """
        self.debug('visit_ident_expr')
        return self.env[expr.ident]

    def debug(self, msg):
        if self.enable_debug:
            print(f'[debug:interp] {msg}')

    def run(self, program):
        """Evaluate and print results of TyCalc expressions.

        This is the high-level entry point that takes a textual TyCalc program
        and evaluates it.
        """
        scanner = Scanner(program, enable_debug=self.enable_debug)
        tokens = scanner.scan()
        self.debug(f'Scanner result: {tokens}')

        parser = Parser(tokens, enable_debug=self.enable_debug)
        exprs = parser.parse()
        self.debug(f'Parser result:')
        for expr in exprs:
            self.debug(f' {expr}')

        TypeChecker(exprs, enable_debug=self.enable_debug).check()
        error = False
        for expr in exprs:
            start, end = expr.span()
            exprtext = program[start:end]
            print(f'{exprtext}')
            if typechecking.is_err((ty := typechecking.get_type(expr))):
                print(f'\t[error] {ty} - in expr at pos [{start}, {end})')
            else:
                print(f'\t[{ty}] {self.evaluate(expr)}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'interp',
        description='Interpret TyCalc programs',
    )
    argparser.add_argument('-i', '--input')
    argparser.add_argument('-d', '--debug', action='store_true')
    args = argparser.parse_args()

    interp = Interp(debug=args.debug)

    if args.input is not None:
        with open(args.input, 'rb') as fp:
            progtext = fp.read().decode('utf-8').strip()
        interp.run(progtext)
    else: # Start REPL
        while True:
            s = input('> ')
            print('{}'.format(interp.evaluate(s)))