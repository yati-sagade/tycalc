"""
TyCalc {
  Prog = Stmt*
  Stmt = Expr ";"
  Expr = Assignment
  Assignment = AssignmentAux | LogicalOr
  AssignmentAux = ident "=" Assignment
  LogicalOr = LogicalAnd ("||" LogicalAnd)*
  LogicalAnd = Equality ("&&" Equality)*
  Equality = Comparison (("!=" | "==") Comparison)*
  Comparison = Term (("<=" | "<" | ">" | ">=") Term)*
  Term = Factor (("+" | "-") Factor)*
  Factor = Unary (("*" | "/" | "%") Unary)*
  Unary = ("+" | "-")? Exponent
  Exponent = ExponentExpr | Primary
  ExponentExpr = Exponent ("**" Exponent)*
  Primary = ParenExpr | ident | numlit | boollit
  ParenExpr = "(" Expr ")"
  ident = letter alnum*
  numlit = fractional | integral
  fractional = digit* "." digit+
  integral = digit+
  boollit = "true" | "false"
}
"""
import sys
from typing import Union, Tuple, Any
from abc import abstractmethod, ABCMeta


class Err:

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f'Err({self.msg})'


class TokenBase(metaclass=ABCMeta):

    def __init__(self, lexeme, value, location: int):
        self.lexeme = lexeme
        self.value = value
        # Index into the original program string that marks the
        # start of this token.
        self.location = location

    def __str__(self):
        return f'{self.__class__.__name__}({self.value})'

    __repr__ = __str__

    def __eq__(self, other):
        return self.value == other.value


class Ident(TokenBase):

    def __init__(self, name: str, location: int):
        super().__init__(name, name, location)

    def __hash__(self) -> int:
        return self.value.__hash__()


class Int(TokenBase):

    def __init__(self, lexeme: str, value: int, location: int):
        super().__init__(lexeme, value, location)


class Float(TokenBase):

    def __init__(self, lexeme: str, value: float, location: int):
        super().__init__(lexeme, value, location)


class Bool(TokenBase):

    def __init__(self, lexeme: str, value: bool, location: int):
        super().__init__(lexeme, value, location)


class Sym(TokenBase):

    def __init__(self, value: str, location: int = None):
        super().__init__(value, value, location)


# sorting by len reversed makes it easy to just iterate and match the current prefix of our
# unprocessed input.
SYMBOLS = [
    '<=',
    '>=',
    '<',
    '>',
    '==',
    '!=',
    '!',
    '=',
    '||',
    '&&',
    '+',
    '-',
    '*',
    '**',
    '/',
    '%',
    ';',
]
SYMBOLS.sort(key=len, reverse=True)


class Scanner:

    def __init__(self, input, enable_debug=False):
        self.input = input
        self.curr = 0
        self.tokens = []
        self.enable_debug = enable_debug

    def eof(self):
        return self.curr >= self.input_len()

    # Must check eof() before calling.
    def curr_char(self):
        return self.input[self.curr]

    def advance(self, by=1):
        if not self.eof():
            self.curr += by

    def input_len(self):
        return len(self.input)

    def scan(self):
        while not self.eof():
            self.tokens.append(self.next_token())
        return self.tokens

    def debug(self, msg):
        if self.enable_debug:
            print('[debug:scanner] ' + msg)

    def next_token(self):
        while not self.eof() and self.input[self.curr].isspace():
            self.curr += 1

        if self.eof():
            return None

        ib = self.maybe_ident_or_bool_literal()
        if ib is not None:
            return ib

        num = self.maybe_number_literal()
        if num is not None:
            return num

        r = self.maybe_operator_or_symbol()
        if r is not None:
            return r

        return Err(
            f'Cannot scan further, remaining input at index {self.curr}: {self.input[self.curr:]}, scanned so far: {self.tokens}'
        )

    def maybe_operator_or_symbol(self):
        for sym in SYMBOLS:
            if self.input[self.curr:].startswith(sym):
                s = Sym(sym, location=self.curr)
                self.advance(by=len(sym))
                return s
        return None

    def maybe_ident_or_bool_literal(self):
        start = self.curr
        end = self.curr
        while not self.eof() and self.curr_char().isalnum():
            if start == self.curr and not self.curr_char().isalpha():
                return None  # First char should be alphabetical.
            self.advance()
        lexeme = self.input[start:self.curr]
        if lexeme == '':
            return None
        if lexeme == 'true':
            return Bool(lexeme, True, start)
        if lexeme == 'false':
            return Bool(lexeme, False, start)
        return Ident(lexeme, start)

    def maybe_number_literal(self):
        self.debug('num?')
        start = self.curr

        def _advance_while_digit():
            saw_digits = False
            while not self.eof() and self.curr_char().isdigit():
                saw_digits = True
                self.advance()
            return saw_digits

        saw_integral_digits = _advance_while_digit()

        # TODO: Support exponent syntax (e.g., 1.2e3 == 1.2E3 == 12000.0)
        if not self.eof() and self.curr_char() == '.':
            self.advance()
            saw_fractional_digits = _advance_while_digit()
            # Allow "1.", ".1", but not just ".".
            if saw_fractional_digits or saw_integral_digits:
                lexeme = self.input[start:self.curr]
                return Float(lexeme, value=float(lexeme), location=start)
        elif saw_integral_digits:
            lexeme = self.input[start:self.curr]
            return Int(lexeme, value=int(lexeme), location=start)
        return None


class Expr(metaclass=ABCMeta):

    def __init__(self):
        self.attrs = {}
    
    @abstractmethod
    def span(self) -> Tuple[int, int]:
      """Return the [start, end) for this expression."""
      pass

    @abstractmethod
    def accept(self, visitor: 'ExprVisitor'):
        pass


class BinaryExpr(Expr):

    def __init__(self, op: Sym, left: Expr, right: Expr):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f'BinaryExpr({self.op}, {self.left}, {self.right}; attrs={self.attrs})'
    
    def span(self):
        start, _  = self.left.span()
        _, end = self.right.span() 
        return (start, end)

    def accept(self, visitor: 'ExprVisitor'):
        return visitor.visit_binary_expr(self)


class UnaryExpr(Expr):

    def __init__(self, op: Sym, arg: Expr):
        super().__init__()
        self.op = op
        self.arg = arg

    def __repr__(self):
        return f'UnaryExpr({self.op}, {self.arg}; attrs={self.attrs})'
    
    def span(self):
        start = self.op.location
        _, end = self.arg.span()
        return (start, end)

    def accept(self, visitor: 'ExprVisitor'):
        return visitor.visit_unary_expr(self)


class Literal(Expr):

    def __init__(self, value: Union[Int, Bool, Float]):
        super().__init__()
        self.value = value

    def __repr__(self):
        return f'Literal({self.value}; attrs={self.attrs})'
    
    def span(self):
        start = self.value.location
        end = start + len(self.value.lexeme)
        return (start, end)

    def accept(self, visitor: 'ExprVisitor'):
        return visitor.visit_literal(self)


class IdentExpr(Expr):

    def __init__(self, ident: Ident):
        super().__init__()
        self.ident = ident

    def accept(self, visitor: 'ExprVisitor'):
        return visitor.visit_ident_expr(self)
    
    def span(self):
        start = self.ident.location
        end = start + len(self.ident.value)
        return (start, end)

    def __repr__(self):
        return f'IdentExpr({self.ident}; attrs={self.attrs})'


class ExprVisitor(metaclass=ABCMeta):

    @abstractmethod
    def visit_binary_expr(self, expr: BinaryExpr) -> Any:
        pass

    @abstractmethod
    def visit_unary_expr(self, expr: UnaryExpr) -> Any:
        pass

    @abstractmethod
    def visit_literal(self, expr: Literal) -> Any:
        pass

    @abstractmethod
    def visit_ident_expr(self, expr: IdentExpr) -> Any:
        pass


class Parser:

    def __init__(self, tokens, enable_debug=False):
        self.enable_debug = enable_debug
        self.curr = 0
        self.tokens = tokens
        self.stmts = []

    def debug(self, msg):
        if self.enable_debug:
            print(f'[debug] {msg}')

    def eof(self):
        return self.curr >= len(self.tokens)

    # Must check eof() before calling.
    def curr_tok(self):
        return self.tokens[self.curr]

    def advance(self, by=1):
        self.curr += by

    def parse(self):
        self.stmts = []
        while not self.eof():
            self.stmts.append(self.stmt())
        return self.stmts

    def stmt(self):
        self.debug('stmt?')
        expr = self.expr()
        self.consume(Sym(';', location=None))
        self.debug(f'parsed stmt {expr}')
        return expr

    def expr(self):
        self.debug('expr?')
        return self.assignment()

    def assignment(self):
        self.debug('assignment?')
        if self.lookahead(Ident, Sym('=', location=None)):
            ident = self.consume(Ident)
            eq = self.consume(Sym('=', location=None))
            rhs = self.assignment()
            self.debug(f'parsed assignment to {ident} of {rhs}')
            return BinaryExpr(eq, IdentExpr(ident), rhs)
        else:
            return self.logical_or()

    def _accumulate_binary_expr_tree(self, parse_fn, *ops):
        expr = parse_fn()
        while (op := self.match_any(*ops)) is not None:
            rhs = parse_fn()
            expr = BinaryExpr(op, expr, rhs)
        return expr

    def logical_or(self):
        self.debug('logical_or?')
        return self._accumulate_binary_expr_tree(self.logical_and,
                                                 Sym('||', location=None))

    def logical_and(self):
        self.debug('logical_and?')
        return self._accumulate_binary_expr_tree(self.equality,
                                                 Sym('&&', location=None))

    def equality(self):
        self.debug('equality?')
        return self._accumulate_binary_expr_tree(self.comparison,
                                                 Sym('==', location=None),
                                                 Sym('!=', location=None))

    def comparison(self):
        self.debug('comparison?')
        return self._accumulate_binary_expr_tree(self.term,
                                                 Sym('<', location=None),
                                                 Sym('<=', location=None),
                                                 Sym('=>', location=None),
                                                 Sym('>', location=None))

    def term(self):
        self.debug('term?')
        return self._accumulate_binary_expr_tree(self.factor,
                                                 Sym('+', location=None),
                                                 Sym('-', location=None))

    def factor(self):
        self.debug('factor?')
        return self._accumulate_binary_expr_tree(self.unary,
                                                 Sym('*', location=None),
                                                 Sym('/', location=None),
                                                 Sym('%', location=None))

    def unary(self):
        self.debug('unary?')
        if (op := self.match_any(Sym('+', location=None), Sym('-',
                                                              location=None),
                                 Sym('!', location=None))) is not None:
            arg = self.exponent()
            return UnaryExpr(op, arg)
        else:
            return self.exponent()

    def exponent(self):
        self.debug('exponent?')
        return self._accumulate_binary_expr_tree(self.primary,
                                                 Sym('**', location=None))

    def primary(self):
        self.debug('primary?')
        if self.match(Sym('(', location=None)) is not None:
            self.debug(' parenexpr?')
            expr = self.expr()
            self.consume(Sym(')', location=None))
            return expr
        elif (ident := self.match(Ident)) is not None:
            return IdentExpr(ident)
        elif (literal := self.match_any(Bool, Int, Float)) is not None:
            return Literal(literal)
        self.bail('Expected expression, identifier or literal')

    def bail(self, msg):
        raise Exception(
            f'Parser error: {msg}; Parsed so far: {self.stmts}; Remaining input: {self.tokens[self.curr:]}'
        )

    def match_any(self, *toks):
        for tok in toks:
            if (m := self.match(tok)) is not None:
                return m
        return None

    def lookahead(self, *toks):
        if self.curr + len(toks) > len(self.tokens):
            return False
        return all(
            self._check(self.tokens[self.curr + i], tok)
            for i, tok in enumerate(toks))

    def _check(self, token, token_or_type):
        if isinstance(token_or_type, type):
            return type(token) == token_or_type
        return token == token_or_type

    def check(self, token_or_type):
        return self._check(self.curr_tok(), token_or_type)

    def match(self, tok):
        if self.eof():
            return None
        if not self.check(tok):
            return None
        ret = self.curr_tok()
        self.advance()
        return ret

    def consume_any(self, *toks):
        m = self.match_any(*toks)
        assert m is not None,\
          f'Expected one of {toks}, parsed so far: {self.stmts},'\
            f'remaining input: {self.tokens[self.curr:]}'
        return m

    def consume(self, tok):
        m = self.match(tok)
        assert m is not None,\
          f'Expected {tok}, parsed so far: {self.stmts},'\
            f'remaining input: {self.tokens[self.curr:]}'
        return m

def _run_file(inputfile):
    with open(inputfile, 'rb') as fp:
        program = fp.read().decode('utf-8').strip()

    scanner = Scanner(program)
    tokens = scanner.scan()

    parser = Parser(tokens)
    exprs = parser.parse()
    print(f'====== Input program =========')
    print(program)
    print(f'====== Parse trees (one line per input expr) =========')
    for expr in exprs:
        print(f' {expr}')

if __name__ == '__main__':
    try:
        inputfile = sys.argv[1]
        _run_file(inputfile)
    except IndexError:
        while True:
            s = input('> ')
            parser = Parser(Scanner(s).scan())
            print('{}'.format(parser.parse()))