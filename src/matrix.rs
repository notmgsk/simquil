use num::complex::Complex64;
use std::collections::HashMap;
use thiserror::Error;

use crate::gates::{standard::gate_matrix, standard::GateError, QGate};

use crate::matrix::InstructionMatrixError::{InvalidInstruction, InvalidQubit};
use quil::expression::EvaluationError;
use quil::instruction::{Instruction, Qubit};

pub const C0: Complex64 = Complex64::new(0.0, 0.0);
pub const C1: Complex64 = Complex64::new(1.0, 0.0);
pub const I1: Complex64 = Complex64::new(0.0, 1.0);

#[derive(Error, Debug)]
pub enum InstructionMatrixError {
    #[error("Invalid parameter")]
    InvalidParameter(#[from] EvaluationError),
    #[error("Cannot create matrix from gate with variable qubit {0}")]
    InvalidQubit(String),
    #[error("Cannot create matrix from non-gate instruction {0}")]
    InvalidInstruction(String),
    #[error("Gate matrix failed")]
    InvalidGateError(#[from] GateError),
}

pub fn instruction_matrix(instruction: Instruction) -> Result<QGate, InstructionMatrixError> {
    match instruction {
        Instruction::Gate {
            name,
            parameters,
            qubits,
            modifiers: _,
        } => {
            let params: Vec<f64> = parameters
                .iter()
                .map(
                    |p| match p.to_owned().evaluate_to_complex(&HashMap::new()) {
                        Ok(c) => Ok(c.re),
                        Err(e) => Err(e),
                    },
                )
                .collect::<Result<Vec<_>, _>>()?;
            let qubits: Vec<usize> = qubits
                .iter()
                .map(|q| match q {
                    Qubit::Fixed(i) => Ok(*i as usize),
                    Qubit::Variable(q) => Err(InvalidQubit(q.clone())),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(gate_matrix(name, params, qubits)?)
        }
        instruction => Err(InvalidInstruction(instruction.to_string())),
    }
}

#[cfg(test)]
pub mod test {
    use crate::matrix::instruction_matrix;
    use quil::expression::Expression;
    use quil::instruction::{Instruction, Qubit};

    #[test]
    pub fn blah() {
        let instruction = Instruction::Gate {
            name: "RX".to_string(),
            parameters: vec![Expression::Variable("yo".to_string())],
            qubits: vec![Qubit::Variable("a".to_string()), Qubit::Fixed(1)],
            modifiers: vec![],
        };
        match instruction_matrix(instruction) {
            Ok(_) => {}
            Err(e) => println!("{:?}", e),
        }
    }
}
