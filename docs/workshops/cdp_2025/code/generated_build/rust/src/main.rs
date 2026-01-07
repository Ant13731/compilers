use std::collections::HashMap;
use std::collections::HashSet;

use bidirectional_map::Bimap;

// Simile types
type Nat = u64;
type Int = i64;
type Float = f64;
type Pair<A, B> = (A, B);
type Relation<A, B> = Bimap<A, B>;

// TODO define custom relation (bidirectional map) type
// - make constructor take in list of pair type/tuple
// - add properties/traits to determine totality, surjectivity, injectivity, etc.
// - select implementations of methods that are most efficient for the relation type (with a fallback in case no type is specified)

//Generated code:

fn main() {
    let mut location: Relation<&str, Int> =
        Relation::from_hash_map(HashMap::from([("SYNT", 100), ("ABC", 200), ("CDP", 300)]));
    let mut attends: Relation<&str, &str> = Relation::from_hash_map(HashMap::from([
        ("Alice", "SYNT"),
        ("Bob", "ABC"),
        ("Charlie", "SYNT"),
    ]));
    let mut room: Int = 100;
    let mut num_meals: Int = 0;
    for (_fresh_var_241584147521002673, _fresh_var_280688665694234642) in attends.iter() {
        if (*location.get_fwd(_fresh_var_280688665694234642).unwrap() == room) {
            num_meals = ((num_meals) + (1));
        }
    }
    println!("{:?}", num_meals);
}
