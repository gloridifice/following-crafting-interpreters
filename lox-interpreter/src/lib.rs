use std::{
    collections::HashMap,
    fmt::format,
    fs::File,
    io::Read,
    path::Path,
    sync::{Arc, LazyLock, Mutex},
    thread::panicking,
};

pub static ERRORS: LazyLock<Arc<Mutex<Vec<anyhow::Error>>>> =
    LazyLock::new(|| Arc::new(Default::default()));

pub static KEYWORDS_MAP: LazyLock<HashMap<&'static str, TokenType>> = LazyLock::new(|| {
    [
        ("and", TokenType::And),
        ("class", TokenType::Class),
        ("else", TokenType::Else),
        ("false", TokenType::False),
        ("for", TokenType::For),
        ("if", TokenType::If),
        ("nil", TokenType::Nil),
        ("or", TokenType::Or),
        ("print", TokenType::Print),
        ("return", TokenType::Return),
        ("super", TokenType::Super),
        ("this", TokenType::This),
        ("var", TokenType::Var),
        ("while", TokenType::While),
    ]
    .into_iter()
    .collect()
});

#[derive(Clone, PartialEq)]
pub enum TokenType {
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Minus,
    Plus,
    Semicolon,
    Slash,
    Star,
    Bang,
    BangEqual,
    Equal,
    EqualEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Identifier,

    // Literals
    String(Arc<String>),
    Number(f64),

    // Keywords
    And,
    Class,
    Else,
    False,
    Fun,
    For,
    If,
    Nil,
    Or,
    Print,
    Return,
    Super,
    This,
    True,
    Var,
    While,
    Eof,
}

pub struct Token {
    token_type: TokenType,
    lexeme: String,
    line: usize,
}

pub struct Scanner {
    source: String,
    tokens: Vec<Token>,
    start: usize,
    current: usize,
    line: usize,
}

impl Token {
    pub fn new(token_type: TokenType, lexeme: String, line: usize) -> Self {
        Self {
            token_type,
            lexeme,
            line,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LiteralValue {
    Nil,
    Number(f64),
    String(Arc<String>),
    Bool(bool),
}

impl Scanner {
    pub fn new(source: String) -> Self {
        Self {
            source,
            tokens: Vec::new(),
            start: 0,
            current: 0,
            line: 1,
        }
    }

    pub fn scan_tokens(&mut self) -> &mut Vec<Token> {
        while !self.is_at_end() {
            self.start = self.current;
            self.scan_tokens();
        }

        self.tokens
            .push(Token::new(TokenType::Eof, String::new(), self.line));

        &mut self.tokens
    }

    fn is_at_end(&self) -> bool {
        self.current as usize >= self.source.len()
    }

    #[allow(unused)]
    fn scan_token(&mut self) {
        let c = self.advance();
        match c {
            '(' => self.add_token(TokenType::LeftParen),
            ')' => self.add_token(TokenType::RightParen),
            '{' => self.add_token(TokenType::LeftBrace),
            '}' => self.add_token(TokenType::RightBrace),
            ',' => self.add_token(TokenType::Comma),
            '.' => self.add_token(TokenType::Dot),
            '-' => self.add_token(TokenType::Minus),
            '+' => self.add_token(TokenType::Plus),
            ';' => self.add_token(TokenType::Semicolon),
            '*' => self.add_token(TokenType::Star),
            '!' => {
                if !self.peek_match_and_try_add_token('=', TokenType::BangEqual) {
                    self.add_token(TokenType::Bang);
                }
            }
            '=' => {
                if !self.peek_match_and_try_add_token('=', TokenType::EqualEqual) {
                    self.add_token(TokenType::Equal);
                }
            }
            '<' => {
                if !self.peek_match_and_try_add_token('=', TokenType::LessEqual) {
                    self.add_token(TokenType::Less);
                }
            }
            '>' => {
                if !self.peek_match_and_try_add_token('=', TokenType::GreaterEqual) {
                    self.add_token(TokenType::GreaterEqual);
                }
            }
            '/' => {
                if self.peek_match('/') {
                    while self.peek().is_some_and(|it| it != '\n') {
                        self.advance();
                    }
                } else {
                    self.add_token(TokenType::Slash);
                }
            }

            ' ' | '\r' | '\t' => {} //Ignore whitespace

            '\n' => {
                self.line += 1;
            }

            '"' => self.string(),

            //TODO!
            _ => {
                if c.is_digit(10) {
                    self.number();
                } else if c.is_alphabetic() {
                    self.identifier();
                } else {
                    ERRORS
                        .lock()
                        .unwrap()
                        .push(anyhow::anyhow!("Unexcepted character <{}>!", c))
                }
            }
        }
    }

    fn identifier(&mut self) {
        while self
            .peek()
            .is_some_and(|it| Self::is_alphabetic_or_numeric(it))
        {
            self.advance();
        }

        let type_ = KEYWORDS_MAP
            .get(&self.source[self.start..self.current])
            .cloned()
            .unwrap_or(TokenType::Identifier);

        self.add_token(type_);
    }

    fn is_alphabetic_or_numeric(c: char) -> bool {
        c.is_digit(10) || c.is_alphabetic()
    }

    fn number(&mut self) {
        while self.peek().is_some_and(|it| it.is_digit(10)) {
            self.advance();
        }

        if self
            .peek()
            .is_some_and(|it| it == '.' && self.peek_next().is_some_and(|it| it.is_digit(10)))
        {
            self.advance();

            while self.peek().is_some_and(|it| it.is_digit(10)) {
                self.advance();
            }
        }

        self.add_token(TokenType::Number(
            self.source[self.start..self.current].parse().unwrap(),
        ));
    }

    fn string(&mut self) {
        while self.peek().is_some_and(|it| it != '"') {
            if self.peek().is_some_and(|it| it == '\n') {
                self.line += 1;
            }

            self.advance();
        }

        if self.is_at_end() {
            ERRORS.lock().unwrap().push(anyhow::anyhow!(
                "Unterminated string, at line {}",
                self.line
            ));
            return;
        }

        self.advance();

        let value = self.source[(self.start + 1)..(self.current + 1)].to_string();
        self.add_token(TokenType::String(Arc::new(value)));
    }

    fn advance(&mut self) -> char {
        let ret = self.source.chars().nth(self.current).unwrap();
        self.current += 1;
        ret
    }

    fn add_token(&mut self, type_: TokenType) {
        let text = self.source[self.start..self.current].to_string();
        self.tokens.push(Token::new(type_, text, self.line));
    }

    fn peek(&self) -> Option<char> {
        self.char_at(self.current)
    }

    fn peek_next(&self) -> Option<char> {
        self.char_at(self.current + 1)
    }

    fn char_at(&self, pos: usize) -> Option<char> {
        self.source.chars().nth(pos)
    }

    fn peek_match(&mut self, expected: char) -> bool {
        if self.peek().is_some_and(|it| it == expected) {
            self.current += 1;
            true
        } else {
            false
        }
    }

    fn peek_match_and_try_add_token(&mut self, expected: char, type_: TokenType) -> bool {
        if self.peek().is_some_and(|it| it == expected) {
            self.add_token(type_);
            self.current += 1;
            true
        } else {
            false
        }
    }
}

pub enum Expr {
    Binary {
        left: Arc<Expr>,
        operator: Arc<Token>,
        right: Arc<Expr>,
    },

    Unary {
        operator: Arc<Token>,
        right: Arc<Expr>,
    },

    Literal(LiteralValue),
    Grouping(Arc<Expr>),
}

pub trait Visitor<T> {
    fn visit(&self, expr: &Expr) -> T;
}

struct AstPrinter;

impl Visitor<String> for AstPrinter {
    fn visit(&self, expr: &Expr) -> String {
        match expr {
            Expr::Binary {
                left,
                operator,
                right,
            } => format!(
                "({} {} {})",
                &operator.lexeme,
                self.visit(&left),
                self.visit(&right)
            ),
            Expr::Unary { operator, right } => {
                format!("({} {})", &operator.lexeme, self.visit(&right))
            }
            Expr::Literal(token) => format!("{:?}", token),
            Expr::Grouping(expr) => format!("(group {})", self.visit(&expr)),
        }
    }
}

pub struct Parser {
    tokens: Vec<Arc<Token>>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Arc<Token>>) -> Self {
        Self { tokens, current: 0 }
    }

    fn expression(&mut self) -> Expr {
        return self.equality();
    }

    fn equality(&mut self) -> Expr {
        let mut expr = self.comparison();

        while self.peek_match(&[TokenType::BangEqual, TokenType::EqualEqual]) {
            let operator = self.previous();
            let right = self.comparison();
            expr = Expr::Binary {
                left: Arc::new(expr),
                operator,
                right: Arc::new(right),
            }
        }

        return expr;
    }

    fn comparison(&mut self) -> Expr {
        let mut expr = self.term();
        use TokenType::*;
        while self.peek_match(&[Greater, GreaterEqual, Less, LessEqual]) {
            let operator = self.previous();
            let right = self.term();
            expr = Expr::Binary {
                left: Arc::new(expr),
                operator,
                right: Arc::new(right),
            }
        }
        expr
    }

    fn term(&mut self) -> Expr {
        let mut expr = self.factor();

        use TokenType::*;
        while self.peek_match(&[Minus, Plus]) {
            let operator = self.previous();
            let right = self.factor();
            expr = Expr::Binary {
                left: Arc::new(expr),
                operator,
                right: Arc::new(right),
            }
        }

        expr
    }

    fn factor(&mut self) -> Expr {
        let mut expr = self.unary();
        use TokenType::*;
        while self.peek_match(&[Slash, Star]) {
            let operator = self.previous();
            let right = self.unary();
            expr = Expr::Binary {
                left: Arc::new(expr),
                operator,
                right: Arc::new(right),
            };
        }

        expr
    }

    fn unary(&mut self) -> Expr {
        use TokenType::*;
        if self.peek_match(&[Bang, Minus]) {
            let operator = self.previous();
            let right = self.unary();
            return Expr::Unary {
                operator,
                right: Arc::new(right),
            };
        }

        return self.primary().unwrap();
    }

    fn primary(&mut self) -> Option<Expr> {
        use TokenType::*;

        if let Some(type_) = self.peek().map(|it| it.token_type.clone()) {
            let expr = match type_ {
                False => Expr::Literal(LiteralValue::Bool(false)),
                True => Expr::Literal(LiteralValue::Bool(true)),
                Nil => Expr::Literal(LiteralValue::Nil),
                Number(v) => Expr::Literal(LiteralValue::Number(v)),
                String(v) => Expr::Literal(LiteralValue::String(v.clone())),
                LeftParen => {
                    let expr = self.expression();
                    self.consume(RightParen, "Expect ')' after expression.");
                    Expr::Grouping(Arc::new(expr))
                }
                _ => return None,
            };

            return Some(expr);
        }

        None
    }

    fn consume(&mut self, type_: TokenType, message: &'static str) -> Arc<Token> {
        if self.peek().is_some_and(|it| it.token_type == type_) {
            self.advance().unwrap()
        } else {
            panic!("{message}")
        }
    }

    fn advance(&mut self) -> Option<Arc<Token>> {
        let ret = self.tokens.get(self.current);
        self.current += 1;
        ret.cloned()
    }

    fn previous(&self) -> Arc<Token> {
        self.tokens.get(self.current - 1).unwrap().clone()
    }

    fn peek(&self) -> Option<Arc<Token>> {
        self.tokens.get(self.current).cloned()
    }

    fn peek_match(&mut self, patterns: &[TokenType]) -> bool {
        if self
            .peek()
            .is_some_and(|peek| patterns.iter().any(|it| *it == peek.token_type))
        {
            self.advance();
            true
        } else {
            false
        }
    }
}

pub fn run_file(path: impl AsRef<Path>) -> anyhow::Result<()> {
    let mut string = String::new();
    File::open(path)?.read_to_string(&mut string)?;

    run(&string)?;

    Ok(())
}

pub fn run(source: &str) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ast() {
        // let expr = Expr::Grouping(Arc::new(Expr::Binary {
        //     left: Arc::new(Expr::Unary {
        //         operator: Token::new(TokenType::Minus, "-".to_string(), 1),
        //         right: Arc::new(Expr::Literal(Token::new(
        //             TokenType::Number(12.0),
        //             "12".to_string(),
        //             2,
        //         ))),
        //     }),
        //     operator: Token::new(TokenType::Star, "*".to_string(), 1),
        //     right: Arc::new(Expr::Literal(Token::new(
        //         TokenType::Number(1.0),
        //         "3".to_string(),
        //         1,
        //     ))),
        // }));
        // let a = AstPrinter;
        // println!("{}", a.visit(&expr));
    }
}
