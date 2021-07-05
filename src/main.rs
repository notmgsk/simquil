use itertools::Itertools;
use quil::{
    instruction::{Instruction, Qubit},
    program::Program,
};
use std::{
    collections::HashSet,
    fs,
    io::{self, Read},
    str::FromStr,
};
use structopt::StructOpt;
use vm::VM;

pub mod gates;
pub mod matrix;
pub mod vm;
pub mod wavefunction;

#[derive(StructOpt)]
struct Cli {
    quil_file: Option<String>,
}

fn run(cli: Cli) {
    let quil = match cli.quil_file {
        Some(path) => fs::read_to_string(path).expect("bad read"),
        None => {
            let mut stdin = io::stdin();
            let mut buf = String::new();
            stdin.read_to_string(&mut buf).expect("bad read from stdin");
            buf
        }
    };
    let program = Program::from_str(&quil).expect("bad program");
    let used_qubits = qubits_in_program(&program);
    let max_qubits_needed = used_qubits.iter().max().unwrap() + 1;

    let mut vm = VM::new(max_qubits_needed, program);

    vm.run().ok();
    println!("Wavefunction amplitudes:");
    println!();
    println!("{:?}", vm);

    println!("Classical memory:");
    println!();
    vm.memory
        .iter()
        .sorted_by_key(|x| x.0)
        .for_each(|(mref, data)| println!("{}[0..{}]:\t {:?}", mref, data.len(), data));
}

fn main() {
    env_logger::init();
    run(Cli::from_args())
}

/// Get all the (numeric) qubits used in the program
fn qubits_in_program(program: &Program) -> Vec<u64> {
    let mut used_qubits: HashSet<u64> = HashSet::new();

    program.instructions.iter().for_each(|i| match i {
        Instruction::Gate { qubits, .. } => qubits.iter().for_each(|q| {
            if let Qubit::Fixed(i) = q {
                used_qubits.insert(*i);
            }
        }),
        Instruction::Measurement {
            qubit: Qubit::Fixed(i),
            ..
        } => {
            used_qubits.insert(*i);
        }
        Instruction::Reset {
            qubit: Some(Qubit::Fixed(i)),
        } => {
            used_qubits.insert(*i);
        }
        Instruction::Delay { qubits, .. } => qubits.iter().for_each(|q| {
            if let Qubit::Fixed(i) = q {
                used_qubits.insert(*i);
            }
        }),
        Instruction::Fence { qubits } => qubits.iter().for_each(|q| {
            if let Qubit::Fixed(i) = q {
                used_qubits.insert(*i);
            }
        }),
        _ => (),
    });

    used_qubits.into_iter().collect()
}
