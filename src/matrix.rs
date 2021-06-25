use std::collections::HashMap;

use crate::gates::{gate_matrix, QGate};
use num::complex::Complex64;
use quil::instruction::{Instruction, Qubit};

pub const C0: Complex64 = Complex64::new(0.0, 0.0);
pub const C1: Complex64 = Complex64::new(1.0, 0.0);
pub const I1: Complex64 = Complex64::new(0.0, 1.0);

pub fn instruction_matrix(instruction: Instruction) -> QGate {
    match instruction {
        Instruction::Gate {
            name,
            parameters,
            qubits,
            modifiers: _,
        } => gate_matrix(
            name,
            parameters
                .iter()
                .map(|p| {
                    p.to_owned()
                        .evaluate_to_complex(&HashMap::new())
                        .expect("bad")
                        .re
                })
                .collect(),
            qubits
                .iter()
                .map(|q| match q {
                    Qubit::Fixed(i) => *i,
                    Qubit::Variable(_) => todo!(),
                })
                .collect(),
        ),
        _ => todo!(),
    }
}
