use std::collections::HashMap;
// use std::collections::HashSet;
// use std::iter;

// use bidirectional_map::Bimap;

use csv::Writer;
use rand::Rng;
use std::fs::File;
use std::time::Instant;

// Simile types
// type Nat = u64;
type Int = i64;
// type Float = f64;
// type Pair<A, B> = (A, B);
// type Relation<A, B> = Bimap<A, B>;

// We can get away with just using HashMap since we know both location and attends are relational functions
type Relation<A, B> = HashMap<A, B>;

// TODO define custom relation (bidirectional map) type
// - make constructor take in list of pair type/tuple
// - add properties/traits to determine totality, surjectivity, injectivity, etc.
// - select implementations of methods that are most efficient for the relation type (with a fallback in case no type is specified)

//Generated code:

#[cfg(debug_assertions)]
const DEBUG: bool = true;

#[cfg(not(debug_assertions))]
const DEBUG: bool = false;

fn make_inputs(size: Int) -> (Relation<String, Int>, Relation<String, String>, Int) {
    let mut rng = rand::thread_rng();

    let workshops: Vec<String> = (0..size).map(|i| format!("Workshop{}", i)).collect();

    let person_to_workshop_ratio = 5;
    let people: Vec<String> = (0..size * person_to_workshop_ratio)
        .map(|i| format!("Person{}", i))
        .collect();

    // Create the bimaps
    let mut location: Relation<String, Int> = Relation::new();
    let mut attends: Relation<String, String> = Relation::new();

    // Random room number
    let room = rng.gen_range(1..size);

    // Fill location map: workshop -> index
    for (i, w) in workshops.iter().enumerate() {
        location.insert(w.clone(), i.try_into().unwrap());
    }

    // Fill attends map: person -> random workshop
    for p in people.iter() {
        let random_workshop = workshops[rng.gen_range(0..workshops.len())].clone();
        // println!("Person {} attends {}", p, random_workshop);
        attends.insert(p.clone(), random_workshop);
    }

    (location, attends, room)
}

fn get_num_meals(
    location: &Relation<String, Int>,
    attends: &Relation<String, String>,
    room: Int,
) -> Int {
    let mut num_meals: Int = 0;
    for (_person, workshop) in attends.iter() {
        if *location.get(workshop).unwrap() == room {
            num_meals += 1;
        }
    }
    num_meals
}

fn main() {
    let file = if DEBUG {
        File::create("../../results/benchmark_runtime_rust_debug.csv").unwrap()
    } else {
        File::create("../../results/benchmark_runtime_rust_release.csv").unwrap()
    };
    // let file = File::create("../../results/benchmark_runtime_rust_debug.csv").unwrap();
    let mut csv_writer = Writer::from_writer(file);

    // for i in 2..100 {
    //     let inputs = make_inputs(i);

    //     // Run get_num_meals 100 times and time it
    //     let start = Instant::now();
    //     for _ in 0..100 {
    //         let _ = get_num_meals(&inputs.0, &inputs.1, inputs.2);
    //     }
    //     let duration = start.elapsed();

    //     csv_writer
    //         .write_record(&[i.to_string(), duration.as_secs_f64().to_string()])
    //         .unwrap();
    //     println!("{:}, {:}", i, duration.as_secs_f64());
    // }

    for exp in 2..15 {
        let size: Int = 2_i64.pow(exp);
        let inputs = make_inputs(size);

        let start = Instant::now();
        for _ in 0..100 {
            let _ = get_num_meals(&inputs.0, &inputs.1, inputs.2);
        }
        let duration = start.elapsed();

        csv_writer
            .write_record(&[size.to_string(), duration.as_secs_f64().to_string()])
            .unwrap();
        println!("{:}, {:}", size, duration.as_secs_f64());
    }
    // let (location, attends, room) = make_inputs(5);
    // let num_meals = get_num_meals(&location, &attends, room);
    // println!("{:?}", num_meals);

    // let mut location: Relation<&str, Int> =
    //     Relation::from_hash_map(HashMap::from([("SYNT", 100), ("ABC", 200), ("CDP", 300)]));
    // let mut attends: Relation<&str, &str> = Relation::from_hash_map(HashMap::from([
    //     ("Alice", "SYNT"),
    //     ("Bob", "ABC"),
    //     ("Charlie", "SYNT"),
    // ]));
    // let mut room: Int = 100;
    // let mut num_meals: Int = 0;
    // for (_fresh_var_241584147521002673, _fresh_var_280688665694234642) in attends.iter() {
    //     if (*location.get_fwd(_fresh_var_280688665694234642).unwrap() == room) {
    //         num_meals = ((num_meals) + (1));
    //     }
    // }
    // println!("{:?}", num_meals);
}
