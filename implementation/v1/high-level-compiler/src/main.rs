use egg::*;
use env_logger;
mod language_definition;

fn main() {
    env_logger::init();

    let my_expression: RecExpr<SymbolLang> = "(foo a b)".parse().unwrap();
    println!("Parsed expression: {}", my_expression);

    let my_enode = SymbolLang::new("bar", vec![]);

    println!("Hello, world!");
}
