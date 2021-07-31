pub mod gates;
pub mod matrix;
pub mod vm;
pub mod wavefunction;

use crate::vm::MemoryContainer;
use pyo3::prelude::*;
use quil::instruction::{Instruction, Qubit};
use quil::program::Program;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;

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

#[pyfunction]
fn simulate(program: String) -> (String, HashMap<String, Vec<i64>>) {
    let program = Program::from_str(program.as_str()).unwrap();
    let used_qubits = qubits_in_program(&program);
    let max_qubits_needed = match used_qubits.iter().max() {
        None => 0,
        Some(n) => *n + 1,
    };
    let mut vm = vm::VM::new(max_qubits_needed, program.clone());
    vm.run();

    let memory: HashMap<String, Vec<i64>> = vm
        .memory
        .into_iter()
        .filter_map(|(k, v)| {
            if let MemoryContainer::Bit(i) = v {
                Some((k, i))
            } else {
                None
            }
        })
        .collect();

    (program.clone().to_string(true), memory)
}

#[pymodule]
fn simquil(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    Ok(())
}
