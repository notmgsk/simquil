//use std::fmt;

use std::{collections::HashMap, fmt, hash::Hash, usize};

use crate::{gates::QGate, matrix::instruction_matrix, wavefunction::Wavefunction};

use quil::{
    instruction::{Instruction, MemoryReference, Qubit, ScalarType},
    program::{MemoryRegion, Program},
};

#[derive(Debug)]
pub enum MemoryType {
    Bit(Vec<i64>),
    Int(Vec<i64>),
    Real(Vec<f64>),
}

impl MemoryType {
    pub fn len(&self) -> usize {
        match self {
            MemoryType::Bit(b) => b.len(),
            MemoryType::Int(i) => i.len(),
            MemoryType::Real(r) => r.len(),
        }
    }
}

#[derive(Default)]
pub struct VM {
    pub wavefunction: Wavefunction,
    pub memory: HashMap<String, MemoryType>,
    pub program: quil::program::Program,
    pub pc: usize,
    pub n_qubits: u64,
}

impl VM {
    pub fn new(n_qubits: u64, program: Program) -> Self {
        let wavefunction = Wavefunction::ground_state_wavefunction(n_qubits);
        let memory: HashMap<String, MemoryType> = program
            .memory_regions
            .iter()
            .map(|(k, v)| {
                let mem = match v.size.data_type {
                    ScalarType::Bit => MemoryType::Bit(vec![0; v.size.length as usize]),
                    ScalarType::Integer => MemoryType::Int(vec![0; v.size.length as usize]),
                    ScalarType::Real => MemoryType::Real(vec![0.0; v.size.length as usize]),
                    ScalarType::Octet => todo!(),
                };

                (k.clone(), mem)
            })
            .into_iter()
            .collect();
        let mut vm = VM {
            wavefunction,
            memory,
            program,
            n_qubits,
            ..Default::default()
        };
        vm.reset();
        vm
    }

    /// Apply a gate to the wavefunction
    pub fn apply(&mut self, gate: &QGate) {
        self.wavefunction.apply(gate);
    }

    /// Measure a qubit storing the result into the target memory
    pub fn measure(&mut self, qubit: &Qubit, target: MemoryReference) {
        let memory_region = self.memory.get_mut(&target.name).unwrap();

        match qubit {
            Qubit::Fixed(idx) => {
                let measured_value = self.wavefunction.measure(*idx);
                match memory_region {
                    MemoryType::Bit(b) => b[target.index as usize] = measured_value as i64,
                    MemoryType::Int(i) => i[target.index as usize] = measured_value as i64,
                    MemoryType::Real(r) => r[target.index as usize] = measured_value as f64,
                }
            }
            Qubit::Variable(_) => todo!(),
        }
    }

    /// Measure a qubit discarding the result
    pub fn measure_discard(&mut self, qubit: &Qubit) {
        match qubit {
            Qubit::Fixed(idx) => {
                let r = rand::random::<f64>();
                let excited_prob = self.wavefunction.excited_state_probability(*idx);
                let collapsed_state = if r <= excited_prob { 1 } else { 0 };
                self.wavefunction
                    .collapse_wavefunction(*idx, excited_prob, collapsed_state);
            }
            Qubit::Variable(_) => todo!(),
        }
    }

    /// Step forward through program, applying the next instruction
    pub fn step(&mut self) {
        if self.pc >= self.program.instructions.len() {
            return;
        }

        let instruction = self.program.instructions[self.pc].to_owned();

        match &instruction {
            Instruction::Gate {
                name: _,
                parameters: _,
                qubits,
                modifiers: _,
            } => {
                let _qubit = match qubits[0] {
                    Qubit::Fixed(i) => i,
                    Qubit::Variable(_) => todo!(),
                };
                let matrix = instruction_matrix(instruction);
                //let matrix = lift_gate_matrix(matrix, self.n_qubits.try_into().unwrap(), qubit);
                self.apply(&matrix);
            }
            Instruction::Measurement { qubit, target } => match target {
                Some(memref) => self.measure(qubit, memref.to_owned()),
                None => self.measure_discard(qubit),
            },
            _ => (),
        }

        self.pc += 1;
    }

    /// Run the program in its entirety
    pub fn run(&mut self) {
        while self.pc < self.program.instructions.len() {
            self.step();
        }
    }

    /// Reset the VM to its initial state
    pub fn reset(&mut self) {
        self.pc = 0;
        self.wavefunction = Wavefunction::ground_state_wavefunction(self.n_qubits);
    }
}

impl fmt::Debug for VM {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.wavefunction)
    }
}
