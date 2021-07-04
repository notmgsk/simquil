//use std::fmt;

use std::{collections::HashMap, fmt, usize};

use crate::{gates::QGate, matrix::instruction_matrix, wavefunction::Wavefunction};

use quil::{
    instruction::{Instruction, MemoryReference, Qubit, ScalarType},
    program::Program,
};

#[derive(Debug)]
pub enum MemoryContainer {
    Bit(Vec<i64>),
    Int(Vec<i64>),
    Real(Vec<f64>),
}

impl MemoryContainer {
    pub fn len(&self) -> usize {
        match self {
            MemoryContainer::Bit(b) => b.len(),
            MemoryContainer::Int(i) => i.len(),
            MemoryContainer::Real(r) => r.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Default)]
pub struct VM {
    pub wavefunction: Wavefunction,
    pub memory: HashMap<String, MemoryContainer>,
    pub program: quil::program::Program,
    pub pc: usize,
    labels: HashMap<String, usize>,
    pub n_qubits: u64,
}

impl VM {
    pub fn new(n_qubits: u64, program: Program) -> Self {
        let wavefunction = Wavefunction::ground_state_wavefunction(n_qubits);
        let memory: HashMap<String, MemoryContainer> = program
            .memory_regions
            .iter()
            .map(|(k, v)| {
                let mem = match v.size.data_type {
                    ScalarType::Bit => MemoryContainer::Bit(vec![0; v.size.length as usize]),
                    ScalarType::Integer => MemoryContainer::Int(vec![0; v.size.length as usize]),
                    ScalarType::Real => MemoryContainer::Real(vec![0.0; v.size.length as usize]),
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
                    MemoryContainer::Bit(b) => b[target.index as usize] = measured_value as i64,
                    MemoryContainer::Int(i) => i[target.index as usize] = measured_value as i64,
                    MemoryContainer::Real(r) => r[target.index as usize] = measured_value as f64,
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

    /// Skip past the next instruction without changing VM state
    pub fn skip(&mut self) {
        if self.pc >= self.program.instructions.len() {
            return;
        }

        self.pc += 1
    }

    /// Jump to the target if the source memory value is equal to the condition value
    fn jump_with_condition(&mut self, target: &str, source: &MemoryReference, condition: i64) {
        if let Some(MemoryContainer::Bit(bit)) = self.memory.get(&source.name) {
            let val = bit[source.index as usize];

            if val == condition {
                self.jump(target)
            }
        }
    }

    /// Jump (unconditionally) to the target
    fn jump(&mut self, target: &str) {
        // Advance immediately if we know the target position
        if let Some(loc) = self.labels.get(target) {
            log::debug!("Jumping (immediately) to PC {}", *loc);
            self.pc = *loc;
            return;
        }

        // Otherwise, skip through instructions, noting each new
        // label until we find the target
        loop {
            let instruction = &self.program.instructions[self.pc];
            if let Instruction::Label(label) = instruction {
                self.labels.insert(label.clone(), self.pc);
                if label == target {
                    log::debug!("Jumping (slowly) to PC {}", self.pc);
                    break;
                }
            }
            self.pc += 1
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
                self.apply(&matrix);
            }

            Instruction::Measurement { qubit, target } => match target {
                Some(memref) => self.measure(qubit, memref.to_owned()),
                None => self.measure_discard(qubit),
            },

            Instruction::Jump { target } => self.jump(target),

            Instruction::JumpWhen { target, condition } => {
                self.jump_with_condition(target, condition, 1)
            }

            Instruction::JumpUnless { target, condition } => {
                self.jump_with_condition(target, condition, 0)
            }

            Instruction::Label(label) => {
                self.labels.insert(label.clone(), self.pc);
            }

            instruction => {
                log::info!("Ignoring instruction {}", instruction)
            }
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

#[cfg(test)]
mod test {
    use std::str::FromStr;

    use quil::program::Program;

    use super::VM;

    #[test]
    fn test_measure_discard() {
        let program = Program::from_str("MEASURE 0").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();
        assert_eq!(vm.wavefunction.excited_state_probability(0), 0.0);

        let program = Program::from_str("X 0; MEASURE 0").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();
        assert_eq!(vm.wavefunction.excited_state_probability(0), 1.0);
    }

    #[test]
    fn test_measure() {
        let program = Program::from_str("DECLARE ro BIT[1]; MEASURE 0 ro[0]").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();
        assert_eq!(vm.wavefunction.excited_state_probability(0), 0.0);
        match vm.memory.get("ro").unwrap() {
            crate::vm::MemoryContainer::Bit(bit) => {
                assert_eq!(bit.len(), 1);
                assert_eq!(bit[0], 0);
            }
            mtype => panic!("expected MemoryType::BIT, got {:?}", mtype),
        }

        let program = Program::from_str("DECLARE ro BIT[1]; X 0; MEASURE 0 ro[0]").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();
        assert_eq!(vm.wavefunction.excited_state_probability(0), 1.0);

        match vm.memory.get("ro").unwrap() {
            crate::vm::MemoryContainer::Bit(bit) => {
                assert_eq!(bit.len(), 1);
                assert_eq!(bit[0], 1);
            }
            mtype => panic!("expected MemoryType::BIT, got {:?}", mtype),
        }
    }
}
