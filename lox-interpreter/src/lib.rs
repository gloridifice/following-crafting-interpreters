use std::{
    fs::File,
    io::Read,
    path::Path,
    sync::{Arc, LazyLock, Mutex},
};

pub static ERRORS: LazyLock<Arc<Mutex<Vec<anyhow::Error>>>> =
    LazyLock::new(|| Arc::new(Default::default()));

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
    String,
    Number,

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
            //TODO!
            _ => ERRORS
                .lock()
                .unwrap()
                .push(anyhow::anyhow!("Unexcepted character <{}>!", c)),
        }
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.is_at_end() {
            return false;
        };
        if self.source.chars().nth(self.current).unwrap() != expected {
            return false;
        };

        self.current += 1;
        true
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
