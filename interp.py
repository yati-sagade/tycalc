import sys
from parsing import Scanner, Parser, ExprVisitor, BinaryExpr, UnaryExpr, Literal, IdentExpr, Expr, Bool
import typechecking
from typechecking import TypeChecker, get_type, is_err


class Val:
    pass

class IntVal(Val):
    def __init__(self, val: int):
        self.val = val
    
    def __repr__(self):
        return repr(self.val)
    
class FloatVal(Val):
    def __init__(self, val: float):
        self.val = val
    
    def __repr__(self):
        return repr(self.val)

class BoolVal(Val):
    def __init__(self, val: bool):
        self.val = val
    
    def __repr__(self):
        return 'true' if self.val else 'false'

def make(ty: typechecking.Type, val: Val):
    if ty == typechecking.Int():
      return IntVal(val)
    if ty == typechecking.Float():
      return FloatVal(val)
    if ty == typechecking.Bool():
      return BoolVal(val)
    raise Exception(f'Unknown type {ty}')


def _add(l: Val, r: Val, ty: typechecking.Type):
    return make(ty, l.val + r.val)

def _sub(l: Val, r: Val, ty: typechecking.Type):
    return make(ty, l.val - r.val)

def _div(l: Val, r: Val, ty: typechecking.Type):
    return make(ty, l.val / r.val)

def _mul(l: Val, r: Val, ty: typechecking.Type):
    return make(ty, l.val * r.val)

def _mod(l: Val, r: Val, ty: typechecking.Type):
    return make(ty, l.val % r.val)

def _exp(l: Val, r: Val, ty: typechecking.Type):
    return make(ty, l.val ** r.val)

def _logic_and(l: Val, r: Val, ty):
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val and r.val)

def _logic_or(l: Val, r: Val, ty: typechecking.Type):
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val or r.val)

def _cmp_lt(l: Val, r: Val, ty: typechecking.Type):
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val < r.val)

def _cmp_le(l: Val, r: Val, ty: typechecking.Type):
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val <= r.val)

def _cmp_gt(l: Val, r: Val, ty: typechecking.Type):
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val > r.val)

def _cmp_ge(l: Val, r: Val, ty: typechecking.Type):
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(l.val >= r.val)

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
    return make(ty, x.val)

def _unary_minus(x: Val, ty: typechecking.Type):
    return make(ty, -x.val)

def _logic_not(x: BoolVal, ty:typechecking.Type):
    assert typechecking.is_bool(ty), f'Expected bool but got {ty}'
    return BoolVal(not x.val)

UNARY_OPS = {
  '+': _unary_plus,
  '-': _unary_minus,
  '!': _logic_not,
}


class Interp(ExprVisitor):

    def __init__(self, debug=False):
        self.enable_debug = debug
        self.env = {}
    
    def evaluate(self, expr: Expr) -> Val:
        return expr.accept(self)

    def visit_binary_expr(self, expr: BinaryExpr):
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
        self.debug('visit_unary_expr')
        arg = self.evaluate(expr.arg)
        return UNARY_OPS[expr.op.value](arg, get_type(expr))

    def visit_literal(self, expr: Literal):
        self.debug('visit_literal')
        return make(get_type(expr), expr.value.value)

    def visit_ident_expr(self, expr: IdentExpr):
        self.debug('visit_ident_expr')
        return self.env[expr.ident]

    def debug(self, msg):
        if self.enable_debug:
            print(f'[debug:interp] {msg}')

    def run(self, program):
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
                print(
                    f'\t[error] {ty} - in expr at pos [{start}, {end})'
                )
            else:
                print(f'\t[{ty}] {self.evaluate(expr)}')


debug = False
if __name__ == '__main__':
    interp = Interp(debug=debug)
    try:
        inputfile = sys.argv[1]
        with open(inputfile, 'rb') as fp:
            progtext = fp.read().decode('utf-8').strip()
        interp.run(progtext)
    except IndexError:
        while True:
            s = input('> ')
            print('{}'.format(interp.evaluate(s)))