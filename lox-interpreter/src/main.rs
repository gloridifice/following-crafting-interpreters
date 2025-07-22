use lox_interpreter::run;

fn main() {
    run(include_str!("test.lox")).unwrap();
}
