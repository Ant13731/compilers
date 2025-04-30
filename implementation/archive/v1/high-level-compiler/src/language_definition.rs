use egg::*;

// Rewrite rules must be S-expressions (basically lisp syntax)
define_language! {
    enum HighLevelLang {
        Int(i64),
        Bool(bool),
        // Float(f64),
        String(String),
        Name(Symbol),
        Type_(Id),


    }
}
