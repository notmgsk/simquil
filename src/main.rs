use itertools::Itertools;
use std::{
    collections::HashSet,
    fs,
    io::{self, Read},
    str::FromStr,
};
use structopt::StructOpt;

use anyhow::Result;
use thiserror::Error;

use quil::{
    instruction::{Instruction, Qubit},
    program::Program,
};

use vm::VM;

pub mod gates;
pub mod matrix;
pub mod vm;
pub mod wavefunction;

#[derive(Error, Debug)]
pub enum SimquilError {
    #[error("Failed to execute program")]
    ExecutionError(#[from] vm::VMError),
    #[error("Failed to read program")]
    ReadProgramError(#[from] io::Error),
}

#[derive(StructOpt)]
struct Cli {
    quil_file: Option<String>,
}

fn run(cli: Cli) -> Result<()> {
    let quil = match cli.quil_file {
        Some(path) => fs::read_to_string(path).map_err(SimquilError::ReadProgramError)?,
        None => {
            let mut stdin = io::stdin();
            let mut buf = String::new();
            stdin
                .read_to_string(&mut buf)
                .map_err(SimquilError::ReadProgramError)?;
            buf
        }
    };
    let program = Program::from_str(&quil).expect("bad program");
    let used_qubits = qubits_in_program(&program);
    let max_qubits_needed = match used_qubits.iter().max() {
        None => 0,
        Some(n) => *n + 1,
    };

    let mut vm = VM::new(max_qubits_needed, program);

    vm.run().map_err(SimquilError::ExecutionError)?;
    println!("Wavefunction amplitudes:");
    println!();
    println!("{:?}", vm);

    if vm.memory.len() > 0 {
        println!("Classical memory:");
        println!();
        vm.memory
            .iter()
            .sorted_by_key(|x| x.0)
            .for_each(|(mref, data)| println!("{}[0..{}]:\t {:?}", mref, data.len(), data));
    }

    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();

    run(Cli::from_args())
}

/// Get all the (numeric) qubits used in the program
fn qubits_in_program(program: &Program) -> Vec<usize> {
    let mut used_qubits: HashSet<usize> = HashSet::new();

    program.instructions.iter().for_each(|i| match i {
        Instruction::Gate { qubits, .. } => qubits.iter().for_each(|q| {
            if let Qubit::Fixed(i) = q {
                used_qubits.insert(*i as usize);
            }
        }),
        Instruction::Measurement {
            qubit: Qubit::Fixed(i),
            ..
        } => {
            used_qubits.insert(*i as usize);
        }
        Instruction::Reset {
            qubit: Some(Qubit::Fixed(i)),
        } => {
            used_qubits.insert(*i as usize);
        }
        Instruction::Delay { qubits, .. } => qubits.iter().for_each(|q| {
            if let Qubit::Fixed(i) = q {
                used_qubits.insert(*i as usize);
            }
        }),
        Instruction::Fence { qubits } => qubits.iter().for_each(|q| {
            if let Qubit::Fixed(i) = q {
                used_qubits.insert(*i as usize);
            }
        }),
        _ => (),
    });

    used_qubits.into_iter().collect()
}
