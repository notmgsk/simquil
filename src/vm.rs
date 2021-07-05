use std::{collections::HashMap, fmt, io, usize};

use crate::{gates::QGate, matrix::instruction_matrix, wavefunction::Wavefunction};

use quil::instruction::ArithmeticOperand;
use quil::{
    instruction::{Instruction, MemoryReference, Qubit, ScalarType},
    program::Program,
};

#[derive(Debug, Clone)]
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
    ///
    /// A result of Ok(true) indicates that execution is halted, and Ok(false) indicates execution can continue
    pub fn step(&mut self) -> Result<bool, io::Error> {
        if self.pc >= self.program.instructions.len() {
            return Ok(true);
        }

        let instruction = self.program.instructions[self.pc].to_owned();

        match &instruction {
            Instruction::Halt {} => {
                // Do nothing
                return Ok(true);
            }

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

            Instruction::Move {
                destination,
                source,
            } => {
                match destination {
                    ArithmeticOperand::MemoryReference(mref) => {
                        // TODO return an error
                        let target = self
                            .memory
                            .get_mut(&mref.name)
                            .expect("trying to MOVE into an undefined memory region");
                        match target {
                            MemoryContainer::Bit(b) => {
                                b[mref.index as usize] = match source {
                                    ArithmeticOperand::LiteralInteger(i) => *i,
                                    d => {
                                        panic!("expected integer operand for MOVE but got {:?}", d)
                                    }
                                }
                            }
                            MemoryContainer::Int(i) => {
                                i[mref.index as usize] = match source {
                                    ArithmeticOperand::LiteralInteger(i) => *i,
                                    d => {
                                        panic!("expected integer operand for MOVE but got {:?}", d)
                                    }
                                }
                            }
                            MemoryContainer::Real(r) => {
                                r[mref.index as usize] = match source {
                                    ArithmeticOperand::LiteralInteger(i) => *i as f64,
                                    ArithmeticOperand::LiteralReal(r) => *r,
                                    d => {
                                        panic!("expected integer operand for MOVE but got {:?}", d)
                                    }
                                }
                            }
                        }
                    }
                    op => panic!("expected a memory reference but got {:}", op),
                }
            }

            Instruction::Exchange { left, right } => match (left, right) {
                (
                    ArithmeticOperand::MemoryReference(left_mref),
                    ArithmeticOperand::MemoryReference(right_mref),
                ) => {
                    if left_mref.name == right_mref.name {
                        let left_index = left_mref.index;
                        let right_index = right_mref.index;
                        let mem = self.memory.remove(&left_mref.name).unwrap();

                        match mem {
                            MemoryContainer::Bit(mut b) => {
                                b.swap(left_index as usize, right_index as usize);
                                self.memory
                                    .insert(left_mref.name.clone(), MemoryContainer::Bit(b));
                            }
                            MemoryContainer::Int(mut i) => {
                                i.swap(left_index as usize, right_index as usize);
                                self.memory
                                    .insert(left_mref.name.clone(), MemoryContainer::Int(i));
                            }
                            MemoryContainer::Real(mut r) => {
                                r.swap(left_index as usize, right_index as usize);
                                self.memory
                                    .insert(left_mref.name.clone(), MemoryContainer::Real(r));
                            }
                        }
                    } else {
                        let left_memory = self.memory.remove(&left_mref.name).unwrap();
                        let right_memory = self.memory.remove(&right_mref.name).unwrap();

                        match (left_memory, right_memory) {
                                (MemoryContainer::Bit(mut left), MemoryContainer::Bit(mut right)) => {
                                    let tmp = left[left_mref.index as usize];
                                    left[left_mref.index as usize] = right[right_mref.index as usize];
                                    right[right_mref.index as usize] = tmp;

                                    self.memory.insert(left_mref.name.clone(), MemoryContainer::Bit(left));
                                    self.memory.insert(right_mref.name.clone(), MemoryContainer::Bit(right));
                                },
                                (MemoryContainer::Int(mut left), MemoryContainer::Int(mut right)) => {
                                    let tmp = left[left_mref.index as usize];
                                    left[left_mref.index as usize] = right[right_mref.index as usize];
                                    right[right_mref.index as usize] = tmp;

                                    self.memory.insert(left_mref.name.clone(), MemoryContainer::Int(left));
                                    self.memory.insert(right_mref.name.clone(), MemoryContainer::Int(right));
                                },
                                (MemoryContainer::Real(mut left), MemoryContainer::Real(mut right)) => {
                                    let tmp = left[left_mref.index as usize];
                                    left[left_mref.index as usize] = right[right_mref.index as usize];
                                    right[right_mref.index as usize] = tmp;

                                    self.memory.insert(left_mref.name.clone(), MemoryContainer::Real(left));
                                    self.memory.insert(right_mref.name.clone(), MemoryContainer::Real(right));
                                },
                                _ => panic!("cannot EXCHANGE memory locations that are not of the same type ({:?} and {:?})", left_mref, right_mref)
                            }
                    }
                }
                (left, right) => panic!(
                    "expected both operands to be memory references, got {:?} and {:?}",
                    left, right
                ),
            },

            Instruction::Load {
                destination,
                source,
                offset,
            } => {
                let dest_memory = self
                    .memory
                    .remove(&destination.name)
                    .expect("cannot LOAD into an undeclared memory region");
                let source_memory = self
                    .memory
                    .get(source)
                    .expect("cannot LOAD from an undeclared memory region");
                let offset_memory = self
                    .memory
                    .get(&offset.name)
                    .expect("cannot LOAD with offset using an undeclared memory region");

                match (dest_memory, source_memory, offset_memory) {
                    (
                        MemoryContainer::Bit(mut dest_bits),
                        MemoryContainer::Bit(source_ints),
                        MemoryContainer::Int(offset_ints),
                    ) => {
                        dest_bits[destination.index as usize] = source_ints[offset_ints[offset.index as usize] as usize];

                        self.memory.insert(destination.name.clone(),  MemoryContainer::Bit(dest_bits));
                    },
                    (
                        MemoryContainer::Int(mut dest_ints),
                        MemoryContainer::Int(source_ints),
                        MemoryContainer::Int(offset_ints),
                    ) => {
                        dest_ints[destination.index as usize] = source_ints[offset_ints[offset.index as usize] as usize];

                        self.memory.insert(destination.name.clone(),  MemoryContainer::Int(dest_ints));
                    },
                    (
                        MemoryContainer::Real(mut dest_reals),
                        MemoryContainer::Real(source_reals),
                        MemoryContainer::Int(offset_ints),
                    ) => {
                        dest_reals[destination.index as usize] = source_reals[offset_ints[offset.index as usize] as usize];

                        self.memory.insert(destination.name.clone(),  MemoryContainer::Real(dest_reals));
                    }
                    load_instruction => panic!("LOAD destination and source must be of same type, and offset must be of type INTEGER: {:?}", load_instruction)
                }
            }

            Instruction::Store {
                destination,
                offset,
                source,
            } => {
                let dest_memory = self
                    .memory
                    .remove(destination)
                    .expect("cannot STORE into undefined memory region");
                let offset_memory = self
                    .memory
                    .get(&offset.name)
                    .expect("cannot STORE using an undefined offset memory region");

                match (dest_memory, offset_memory, source) {
                    (
                        MemoryContainer::Bit(mut dest_bits),
                        MemoryContainer::Int(offset_ints),
                        ArithmeticOperand::LiteralInteger(i),
                    ) => {
                        let index = offset_ints[offset.index as usize] as usize;
                        dest_bits[index] = *i;

                        self.memory
                            .insert(destination.clone(), MemoryContainer::Bit(dest_bits));
                    }
                    (
                        MemoryContainer::Bit(mut dest_bits),
                        MemoryContainer::Int(offset_ints),
                        ArithmeticOperand::MemoryReference(mref),
                    ) => {
                        let source_memory = self
                            .memory
                            .get(&mref.name)
                            .expect("cannot STORE using an undefined source memory region");
                        match source_memory {
                            MemoryContainer::Bit(source_bits) => {
                                let index = offset_ints[offset.index as usize] as usize;
                                dest_bits[index] = source_bits[mref.index as usize];
                            }
                            sref => panic!("oof"),
                        }

                        self.memory
                            .insert(destination.clone(), MemoryContainer::Bit(dest_bits));
                    }
                    (
                        MemoryContainer::Int(mut dest_ints),
                        MemoryContainer::Int(offset_ints),
                        ArithmeticOperand::LiteralInteger(i),
                    ) => {
                        let index = offset_ints[offset.index as usize] as usize;
                        dest_ints[index] = *i;

                        self.memory
                            .insert(destination.clone(), MemoryContainer::Int(dest_ints));
                    }
                    (
                        MemoryContainer::Int(mut dest_ints),
                        MemoryContainer::Int(offset_ints),
                        ArithmeticOperand::MemoryReference(mref),
                    ) => {
                        let source_memory = self
                            .memory
                            .get(&mref.name)
                            .expect("cannot STORE using an undefined source memory region");
                        match source_memory {
                            MemoryContainer::Int(source_ints) => {
                                let index = offset_ints[offset.index as usize] as usize;
                                dest_ints[index] = source_ints[mref.index as usize];
                            }
                            sref => panic!("oof"),
                        }

                        self.memory
                            .insert(destination.clone(), MemoryContainer::Int(dest_ints));
                    }
                    (
                        MemoryContainer::Real(mut dest_reals),
                        MemoryContainer::Int(offset_ints),
                        ArithmeticOperand::LiteralReal(r),
                    ) => {
                        let index = offset_ints[offset.index as usize] as usize;
                        dest_reals[index] = *r;

                        self.memory
                            .insert(destination.clone(), MemoryContainer::Real(dest_reals));
                    }
                    (
                        MemoryContainer::Real(mut dest_reals),
                        MemoryContainer::Int(offset_ints),
                        ArithmeticOperand::MemoryReference(mref),
                    ) => {
                        let source_memory = self
                            .memory
                            .get(&mref.name)
                            .expect("cannot STORE using an undefined source memory region");
                        match source_memory {
                            MemoryContainer::Real(source_reals) => {
                                let index = offset_ints[offset.index as usize] as usize;
                                dest_reals[index] = source_reals[mref.index as usize];
                            }
                            sref => panic!("oof"),
                        }

                        self.memory
                            .insert(destination.clone(), MemoryContainer::Real(dest_reals));
                    }
                    tup => panic!("{:?}", tup),
                }
            }
            instruction => {
                log::warn!("Ignoring instruction {}", instruction)
            }
        }

        self.pc += 1;

        Ok(false)
    }

    /// Run the program in its entirety
    pub fn run(&mut self) -> Result<bool, io::Error> {
        loop {
            let done = self.step()?;
            if done {
                return Ok(true);
            }
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
    use crate::vm::MemoryContainer;

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

    #[test]
    fn test_halt() {
        let program = Program::from_str("HALT").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();
        assert_eq!(vm.pc, 0);

        let program = Program::from_str("X 0; Y 0; HALT; Z 0").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();
        assert_eq!(vm.pc, 2)
    }

    #[test]
    fn test_move() {
        let program =
            Program::from_str("DECLARE mem REAL[2]; MOVE mem[0] 0.5; MOVE mem[1] 1.0;").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();
        match vm
            .memory
            .get("mem")
            .expect("expected memory region named mem")
        {
            MemoryContainer::Real(r) => {
                assert_eq!(r.len(), 2);
                assert_eq!(r[0], 0.5);
                assert_eq!(r[1], 1.0);
            }
            mcont => panic!("expected a REAL memory region, got {:?}", mcont),
        }
    }

    #[test]
    fn test_exchange() {
        let program =
            Program::from_str("DECLARE mem BIT[2]; MOVE mem[0] 1; EXCHANGE mem[0] mem[1]").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();
        match vm
            .memory
            .get("mem")
            .expect("expected memory region named mem")
        {
            MemoryContainer::Bit(b) => {
                assert_eq!(b.len(), 2);
                assert_eq!(b[0], 0);
                assert_eq!(b[1], 1);
            }
            mcont => panic!("expected a BIT memory region, got {:?}", mcont),
        }

        let program = Program::from_str(
            "DECLARE a REAL[1]; DECLARE b REAL[1]; MOVE b 0.5; EXCHANGE a[0] b[0]",
        )
        .unwrap();
        let mut vm = VM::new(1, program);
        vm.run();
        match vm.memory.get("a").expect("expected memory region named a") {
            MemoryContainer::Real(r) => {
                assert_eq!(r.len(), 1);
                assert_eq!(r[0], 0.5);
            }
            mcont => panic!("expected a REAL memory region, got {:?}", mcont),
        }
        match vm.memory.get("b").expect("expected memory region named b") {
            MemoryContainer::Real(r) => {
                assert_eq!(r.len(), 1);
                assert_eq!(r[0], 0.0);
                println!("{:?}", vm.memory);
            }
            mcont => panic!("expected a REAL memory region, got {:?}", mcont),
        }
    }

    #[test]
    fn test_load() {
        let program = Program::from_str("DECLARE a REAL[2]; DECLARE b REAL[2]; DECLARE n INTEGER[2]; MOVE n[1] 1; MOVE b[1] 1.0; LOAD a[1] b n[1];").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();

        match vm.memory.get("a").expect("expected memory region named a") {
            MemoryContainer::Real(r) => {
                assert_eq!(r.len(), 2);
                assert_eq!(r[1], 1.0);
            }
            mcont => panic!("expected a REAL memory region, got {:?}", mcont),
        }
    }

    #[test]
    fn test_store() {
        let program = Program::from_str("DECLARE a BIT[2]; DECLARE b BIT[2]; DECLARE n INTEGER[2]; MOVE n[1] 1; MOVE b[1] 1; STORE a n[1] b[1];").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();

        match vm.memory.get("a").expect("expected memory region named a") {
            MemoryContainer::Bit(r) => {
                assert_eq!(r.len(), 2);
                assert_eq!(r[1], 1);
            }
            mcont => panic!("expected a REAL memory region, got {:?}", mcont),
        }

        let program = Program::from_str(
            "DECLARE a BIT[2]; DECLARE n INTEGER[2]; MOVE n[1] 1; STORE a n[1] 1;",
        )
        .unwrap();
        let mut vm = VM::new(1, program);
        vm.run();

        match vm.memory.get("a").expect("expected memory region named a") {
            MemoryContainer::Bit(r) => {
                assert_eq!(r.len(), 2);
                assert_eq!(r[1], 1);
            }
            mcont => panic!("expected a REAL memory region, got {:?}", mcont),
        }

        let program = Program::from_str("DECLARE a INTEGER[2]; DECLARE b INTEGER[2]; DECLARE n INTEGER[2]; MOVE n[1] 1; MOVE b[1] 1; STORE a n[1] b[1];").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();

        match vm.memory.get("a").expect("expected memory region named a") {
            MemoryContainer::Int(r) => {
                assert_eq!(r.len(), 2);
                assert_eq!(r[1], 1);
            }
            mcont => panic!("expected a REAL memory region, got {:?}", mcont),
        }

        let program = Program::from_str(
            "DECLARE a INTEGER[2];  DECLARE n INTEGER[2]; MOVE n[1] 1; STORE a n[1] 1;",
        )
        .unwrap();
        let mut vm = VM::new(1, program);
        vm.run();

        match vm.memory.get("a").expect("expected memory region named a") {
            MemoryContainer::Int(r) => {
                assert_eq!(r.len(), 2);
                assert_eq!(r[1], 1);
            }
            mcont => panic!("expected a REAL memory region, got {:?}", mcont),
        }

        let program = Program::from_str("DECLARE a REAL[2]; DECLARE b REAL[2]; DECLARE n INTEGER[2]; MOVE n[1] 1; MOVE b[1] 1.0; STORE a n[1] b[1];").unwrap();
        let mut vm = VM::new(1, program);
        vm.run();

        match vm.memory.get("a").expect("expected memory region named a") {
            MemoryContainer::Real(r) => {
                assert_eq!(r.len(), 2);
                assert_eq!(r[1], 1.0);
            }
            mcont => panic!("expected a REAL memory region, got {:?}", mcont),
        }

        let program = Program::from_str(
            "DECLARE a REAL[2];  DECLARE n INTEGER[2]; MOVE n[1] 1; STORE a n[1] 1.0;",
        )
        .unwrap();
        let mut vm = VM::new(1, program);
        vm.run();

        match vm.memory.get("a").expect("expected memory region named a") {
            MemoryContainer::Real(r) => {
                assert_eq!(r.len(), 2);
                assert_eq!(r[1], 1.0);
            }
            mcont => panic!("expected a REAL memory region, got {:?}", mcont),
        }
    }
}
