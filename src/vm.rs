//use std::fmt;

use std::fmt;

use crate::{gates::QGate, matrix::instruction_matrix, wavefunction::Wavefunction};

use quil::{
    instruction::{Instruction, Qubit},
    program::Program,
};

// 1. Object with a quantum state
// 2. Method to transition state
// 3. Method to apply a quantum gate to a state

pub struct VM {
    wavefunction: Wavefunction,
    program: quil::program::Program,
    pc: usize,
    n_qubits: u64,
}

impl VM {
    pub fn new(n_qubits: u64, program: Program) -> Self {
        let wavefunction = Wavefunction::ground_state_wavefunction(n_qubits);
        let mut vm = VM {
            wavefunction,
            program,
            pc: 0,
            n_qubits,
        };
        vm.reset();
        vm
    }

    pub fn apply(&mut self, gate: &QGate) {
        self.wavefunction.apply(gate);
    }

    pub fn measure(&mut self, qubit: &Qubit) {
        match qubit {
            Qubit::Fixed(idx) => {
                let r = rand::random::<f64>();
                let excited_prob = self.wavefunction.excited_state_probability(idx);
                let collapsed_state = if r <= excited_prob { 1 } else { 0 };
                self.wavefunction
                    .collapse_wavefunction(idx, excited_prob, collapsed_state);
            }
            Qubit::Variable(_) => todo!(),
        }
    }

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
                Some(_) => todo!(),
                None => self.measure(qubit),
            },
            _ => todo!(),
        }

        self.pc += 1;
    }

    pub fn run(&mut self) {
        while self.pc < self.program.instructions.len() {
            self.step();
        }
    }

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
