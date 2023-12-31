"""Typechecking for TyCalc

This module can be used programmatically via the TypeChecker class, but can also be
run directly:

    $ python typechecking.py -i INPUT_FILE

To start a REPL,

    $ python typechecking.py
"""
import sys
import argparse
import abc
import parsing
from collections import namedtuple


class Type:
    """Base class for types."""

    def __repr__(self):
        return f'<type:{type(self).__name__}>'

    def __eq__(self, other):
        return type(self) == type(other)


class Bool(Type):
    """The TyCalc Boolean type."""
    pass


class Int(Type):
    """The TyCalc integral type."""
    pass


class Float(Type):
    """The TyCalc floating point number type."""
    pass


class Err(Type):
    """An "error" type that is assigned to expressions when there is a type-check failure."""

    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return f'<badtype:{self.msg}>'


def is_err(ty):
    return type(ty) == Err


def is_bool(ty):
    return type(ty) == Bool


def is_int(ty):
    return type(ty) == Int


def is_float(ty):
    return type(ty) == Float


def is_numeric(ty):
    return is_int(ty) or is_float(ty)


def coerce(ty1, ty2):
    """Return the "higher" numeric type

    Requires both ty1 and ty2 to be numeric (integral or float) types.

    Returns an Int instance iff. ty1 and ty2 are both integral types, else
    returns a Float instance.
    """
    if not is_numeric(ty1) or not is_numeric(ty2):
        raise ValueError(f'Non-coercible types: {ty1}, {ty2}')
    if is_float(ty1) or is_float(ty2):
        return Float()
    return Int()


TypedExpr = namedtuple('TypedExpr', ('expr', 'type'))


def set_type(expr, ty, strict=True):
    """Set the type attribute in `expr` to the given type in `ty`.
    
    Requires the target expression's type attribute not already be set.
    """
    if strict:
        assert 'type' not in expr.attrs, f'Type already exists for {expr}'
    expr.attrs['type'] = ty


def get_type(expr):
    return expr.attrs['type']


def _arith_result_type(op, lty, rty):
    try:
        return coerce(lty, rty)
    except ValueError:
        return Err(f'Invalid operand types for {op}: {lty}, {rty}')


class TypeChecker(parsing.ExprVisitor):

    def __init__(self, exprs: list[parsing.Expr], enable_debug=False):
        self.exprs = exprs
        self.idents = {}
        self.enable_debug = enable_debug

    def debug(self, msg):
        if self.enable_debug:
            print(f'[debug:tyc] {msg}')

    def check(self):
        for expr in self.exprs:
            self.debug(f'=== Checking {expr}')
            ty = self.compute_type(expr)

    def compute_type(self, expr):
        expr.accept(self)
        return expr.attrs['type']

    def visit_binary_expr(self, expr: parsing.BinaryExpr):
        self.debug('visit_binary_expr')
        if expr.op == parsing.Sym('='):
            assert type(expr.left) == parsing.IdentExpr
            ty = self.compute_type(expr.right)
            self.idents[expr.left.ident] = ty
            if is_err(ty):
                return
            set_type(expr, ty)
        elif expr.op in (parsing.Sym('+'), parsing.Sym('-'), parsing.Sym('*'),
                         parsing.Sym('/'), parsing.Sym('**')):
            lty = self.compute_type(expr.left)
            rty = self.compute_type(expr.right)
            set_type(expr, _arith_result_type(expr.op, lty, rty))
        elif expr.op in (parsing.Sym('||'), parsing.Sym('&&')):
            lty = self.compute_type(expr.left)
            rty = self.compute_type(expr.right)
            if type(lty) == Bool and type(rty) == Bool:
                set_type(expr, Bool())
            else:
                set_type(
                    expr,
                    Err(f'Invalid operands for {expr.op.value}: {lty} and {rty}'
                       ))
        elif expr.op in (parsing.Sym('<'), parsing.Sym('>'), parsing.Sym('<='),
                         parsing.Sym('>=')):
            lty = self.compute_type(expr.left)
            rty = self.compute_type(expr.right)
            # Both args should be numeric for comparison
            if not is_numeric(lty) or not is_numeric(rty):
                set_type(
                    expr,
                    Err(f'Invalid operands for {expr.op.value}: {lty} and {rty}'
                       ))
            else:
                set_type(expr, Bool())

        else:
            raise ValueError(f'Unknown op {expr.op} in {expr}')

    def visit_unary_expr(self, expr: parsing.UnaryExpr):
        self.debug('visit_unary_expr')
        ty = self.compute_type(expr.arg)
        if is_err(ty):
            return
        if expr.op in (parsing.Sym('-'), parsing.Sym('+')):
            if is_numeric(ty):
                set_type(expr, ty)
            else:
                set_type(
                    expr,
                    Err(f'Invalid argtype for unary {expr.op.value}: {ty}'))
        elif expr.op == parsing.Sym('!'):
            if is_bool(ty):
                set_type(expr, Bool())
            else:
                set_type(
                    expr,
                    Err(f'Invalid argtype for unary {expr.op.value}: {ty}'))

    def visit_literal(self, expr: parsing.Literal):
        self.debug(f'visit_literal for {expr}')
        assert type(expr) == parsing.Literal, f'Invalid literal arg {expr}'

        def _literal_type():
            if type(expr.value) == parsing.Int:
                return Int()
            if type(expr.value) == parsing.Float:
                return Float()
            if type(expr.value) == parsing.Bool:
                return Bool()
            raise ValueError(f'Invalid literal {expr}')

        set_type(expr, _literal_type())

    def visit_ident_expr(self, expr: parsing.IdentExpr):
        self.debug('visit_ident')
        set_type(expr, self.idents[expr.ident])


def _run_file(inputfile, debug):
    with open(inputfile, 'rb') as fp:
        program = fp.read().decode('utf-8').strip()

    scanner = parsing.Scanner(program, enable_debug=debug)
    tokens = scanner.scan()

    parser = parsing.Parser(tokens, enable_debug=debug)
    exprs = parser.parse()

    type_checker = TypeChecker(exprs, enable_debug=debug)
    type_checker.check()

    print(f'====== Input program =========')
    print(program)
    print(f'====== Typed parse trees (one line per input expr) =========')
    for expr in exprs:
        print(f' {expr}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'parser',
        description=
        'Parse and typecheck TyCalc programs, and print expression trees with type info',
    )
    argparser.add_argument('-i', '--input')
    argparser.add_argument('-d', '--debug', action='store_true')
    args = argparser.parse_args()

    if args.input is not None:
        _run_file(args.input, args.debug)
    else:
        while True:  # Start REPL
            s = input('> ')
            parser = parsing.Parser(parsing.Scanner(s).scan())
            expr = parser.parse()
            type_checker = TypeChecker(expr)
            type_checker.check()  # Modifies trees to add type attrs
            print('{}'.format(expr))