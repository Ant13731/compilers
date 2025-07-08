fn test(a: str) -> str {
    return f"// Test code for {a}\n"
}
fn main(){

    print(test("code generation"))
    print(test("C++ code generation"))
    print(test("Rust code generation"))
}