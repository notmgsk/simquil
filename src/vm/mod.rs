use std::{collections::HashMap, fmt, usize};

use quil::instruction::ArithmeticOperand;
use quil::{
    instruction::{Instruction, MemoryReference, Qubit, ScalarType},
    program::Program,
};

use crate::{gates::QGate, matrix::instruction_matrix, wavefunction::Wavefunction};

pub mod errors;
pub use errors::*;

#[derive(Debug, Clone)]
pub enum MemoryContainer {
    Bit(Vec<i64>),
    Integer(Vec<i64>),
    Real(Vec<f64>),
}

impl MemoryContainer {
    pub fn len(&self) -> usize {
        match self {
            MemoryContainer::Bit(b) => b.len(),
            MemoryContainer::Integer(i) => i.len(),
            MemoryContainer::Real(r) => r.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn type_of(&self) -> String {
        match self {
            MemoryContainer::Bit(_) => "BIT".to_string(),
            MemoryContainer::Integer(_) => "INTEGER".to_string(),
            MemoryContainer::Real(_) => "REAL".to_string(),
        }
    }

    pub fn bit(self) -> Result<Vec<i64>, VMError> {
        match self {
            MemoryContainer::Bit(v) | MemoryContainer::Integer(v) => Ok(v),
            MemoryContainer::Real(_) => Err(VMError::UnexpectedMemoryRegionType {
                expected: "BIT".to_string(),
                actual: self.type_of(),
            }),
        }
    }

    pub fn integer(self) -> Result<Vec<i64>, VMError> {
        match self {
            MemoryContainer::Bit(b) => Ok(b),
            MemoryContainer::Integer(i) => Ok(i),
            MemoryContainer::Real(_) => Err(VMError::UnexpectedMemoryRegionType {
                expected: "INTEGER".to_string(),
                actual: self.type_of(),
            }),
        }
    }

    pub fn real(self) -> Result<Vec<f64>, VMError> {
        match self {
            MemoryContainer::Real(r) => Ok(r),
            _ => Err(VMError::UnexpectedMemoryRegionType {
                expected: "REAL".to_string(),
                actual: self.type_of(),
            }),
        }
    }
}

#[derive(Default)]
pub struct VM {
    pub wavefunction: Wavefunction,
    pub memory: HashMap<String, MemoryContainer>,
    pub program: quil::program::Program,
    pub pc: usize,
    labels: HashMap<String, usize>,
    pub n_qubits: usize,
}

impl VM {
    pub fn new(n_qubits: usize, program: Program) -> Self {
        let wavefunction = Wavefunction::ground_state_wavefunction(n_qubits);
        let memory: HashMap<String, MemoryContainer> = program
            .memory_regions
            .iter()
            .map(|(k, v)| {
                let mem = match v.size.data_type {
                    ScalarType::Bit => MemoryContainer::Bit(vec![0; v.size.length as usize]),
                    ScalarType::Integer => {
                        MemoryContainer::Integer(vec![0; v.size.length as usize])
                    }
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
    pub fn measure(&mut self, qubit: &Qubit, target: MemoryReference) -> Result<(), VMError> {
        let memory_region =
            self.memory
                .get_mut(&target.name)
                .ok_or(VMError::UndefinedMemoryRegion {
                    name: target.name.clone(),
                })?;

        match qubit {
            Qubit::Fixed(idx) => {
                let measured_value = self.wavefunction.measure(*idx as usize);
                match memory_region {
                    MemoryContainer::Bit(b) => b[target.index as usize] = measured_value as i64,
                    MemoryContainer::Integer(i) => i[target.index as usize] = measured_value as i64,
                    MemoryContainer::Real(r) => r[target.index as usize] = measured_value as f64,
                }
                Ok(())
            }
            Qubit::Variable(v) => Err(VMError::UnresolvedQubitVariable { name: v.clone() }),
        }
    }

    /// Measure a qubit discarding the result
    pub fn measure_discard(&mut self, qubit: &Qubit) -> Result<(), VMError> {
        match qubit {
            Qubit::Fixed(idx) => {
                let r = rand::random::<f64>();
                let excited_prob = self.wavefunction.excited_state_probability(*idx as usize);
                let collapsed_state = if r <= excited_prob { 1 } else { 0 };
                self.wavefunction.collapse_wavefunction(
                    *idx as usize,
                    excited_prob,
                    collapsed_state,
                );
                Ok(())
            }
            Qubit::Variable(v) => Err(VMError::UnresolvedQubitVariable { name: v.clone() }),
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
    fn jump_with_condition(
        &mut self,
        target: &str,
        source: &MemoryReference,
        condition: i64,
    ) -> Result<(), VMError> {
        if let Some(MemoryContainer::Bit(bit)) = self.memory.get(&source.name) {
            let val = bit[source.index as usize];

            if val == condition {
                self.jump(target)?
            }
        }

        Ok(())
    }

    /// Jump (unconditionally) to the target
    fn jump(&mut self, target: &str) -> Result<(), VMError> {
        // Advance immediately if we know the target position
        if let Some(loc) = self.labels.get(target) {
            log::debug!("Jumping (immediately) to PC {}", *loc);
            self.pc = *loc;
            return Ok(());
        }

        // Otherwise, skip through instructions, noting each new
        // label until we find the target
        loop {
            if self.pc >= self.program.instructions.len() {
                return Err(VMError::UndefinedJumpTarget {
                    name: target.to_string(),
                });
            }
            let instruction = &self.program.instructions[self.pc];
            if let Instruction::Label(label) = instruction {
                self.labels.insert(label.clone(), self.pc);
                if label == target {
                    log::debug!("Jumping (slowly) to PC {}", self.pc);
                    return Ok(());
                }
            }
            self.pc += 1
        }
    }

    /// Step forward through program, applying the next instruction
    ///
    /// A result of Ok(true) indicates that execution is halted, and Ok(false) indicates execution can continue
    pub fn step(&mut self) -> Result<bool, VMError> {
        if self.pc >= self.program.instructions.len() {
            return Ok(true);
        }

        let instruction = self.program.instructions[self.pc].to_owned();

        match &instruction {
            Instruction::Halt {} => {
                return Ok(true);
            }

            Instruction::Gate { .. } => {
                // TODO errors
                let matrix = instruction_matrix(instruction);
                self.apply(&matrix);
            }

            Instruction::Measurement { qubit, target } => match target {
                Some(memref) => self.measure(qubit, memref.to_owned())?,
                None => self.measure_discard(qubit)?,
            },

            Instruction::Jump { target } => self.jump(target)?,

            Instruction::JumpWhen { target, condition } => {
                self.jump_with_condition(target, condition, 1)?
            }

            Instruction::JumpUnless { target, condition } => {
                self.jump_with_condition(target, condition, 0)?
            }

            Instruction::Label(label) => {
                if self.labels.insert(label.clone(), self.pc).is_some() {
                    return Err(VMError::DuplicateLabel {
                        name: label.clone(),
                    });
                }
            }

            Instruction::Move {
                destination,
                source,
            } => match destination {
                ArithmeticOperand::MemoryReference(mref) => {
                    let mut dest_mcont =
                        self.memory
                            .remove(&mref.name)
                            .ok_or(VMError::UndefinedMemoryRegion {
                                name: mref.name.clone(),
                            })?;

                    match dest_mcont {
                        MemoryContainer::Bit(ref mut b) => {
                            b[mref.index as usize] = match source {
                                ArithmeticOperand::LiteralInteger(i) => *i,
                                ArithmeticOperand::MemoryReference(m) => self
                                    .memory
                                    .get(&m.name)
                                    .ok_or(VMError::UndefinedMemoryRegion {
                                        name: m.name.clone(),
                                    })?
                                    .clone()
                                    .bit()?[m.index as usize],
                                _ => {
                                    return Err(VMError::UnexpectedLiteralType {
                                        actual: "REAL".to_string(),
                                        expected: "BIT".to_string(),
                                    })
                                }
                            }
                        }
                        MemoryContainer::Integer(ref mut i) => {
                            i[mref.index as usize] = match source {
                                ArithmeticOperand::LiteralInteger(i) => *i,
                                ArithmeticOperand::MemoryReference(m) => self
                                    .memory
                                    .get(&m.name)
                                    .ok_or(VMError::UndefinedMemoryRegion {
                                        name: m.name.clone(),
                                    })?
                                    .clone()
                                    .integer()?[m.index as usize],
                                _ => {
                                    return Err(VMError::UnexpectedLiteralType {
                                        actual: "REAL".to_string(),
                                        expected: "INTEGER".to_string(),
                                    })
                                }
                            }
                        }
                        MemoryContainer::Real(ref mut r) => {
                            r[mref.index as usize] = match source {
                                ArithmeticOperand::LiteralInteger(i) => *i as f64,
                                ArithmeticOperand::LiteralReal(r) => *r,
                                ArithmeticOperand::MemoryReference(m) => self
                                    .memory
                                    .get(&m.name)
                                    .ok_or(VMError::UndefinedMemoryRegion {
                                        name: m.name.clone(),
                                    })?
                                    .clone()
                                    .real()?[m.index as usize],
                            }
                        }
                    }

                    self.memory.insert(mref.name.clone(), dest_mcont);
                }
                op => {
                    return Err(VMError::InvalidMoveDestination {
                        actual: op.to_string(),
                    })
                }
            },

            Instruction::Exchange { left, right } => match (left, right) {
                (
                    ArithmeticOperand::MemoryReference(left_mref),
                    ArithmeticOperand::MemoryReference(right_mref),
                ) => {
                    if left_mref.name == right_mref.name {
                        let left_index = left_mref.index;
                        let right_index = right_mref.index;
                        let mem = self.memory.remove(&left_mref.name).ok_or(
                            VMError::UndefinedMemoryRegion {
                                name: left_mref.name.clone(),
                            },
                        )?;

                        match mem {
                            MemoryContainer::Bit(mut b) => {
                                b.swap(left_index as usize, right_index as usize);
                                self.memory
                                    .insert(left_mref.name.clone(), MemoryContainer::Bit(b));
                            }
                            MemoryContainer::Integer(mut i) => {
                                i.swap(left_index as usize, right_index as usize);
                                self.memory
                                    .insert(left_mref.name.clone(), MemoryContainer::Integer(i));
                            }
                            MemoryContainer::Real(mut r) => {
                                r.swap(left_index as usize, right_index as usize);
                                self.memory
                                    .insert(left_mref.name.clone(), MemoryContainer::Real(r));
                            }
                        }
                    } else {
                        let left_memory = self.memory.remove(&left_mref.name).ok_or(
                            VMError::UndefinedMemoryRegion {
                                name: left_mref.name.clone(),
                            },
                        )?;
                        let right_memory = self.memory.remove(&right_mref.name).ok_or(
                            VMError::UndefinedMemoryRegion {
                                name: right_mref.name.clone(),
                            },
                        )?;

                        match (left_memory, right_memory) {
                            (MemoryContainer::Bit(mut left), MemoryContainer::Bit(mut right)) => {
                                let tmp = left[left_mref.index as usize];
                                left[left_mref.index as usize] = right[right_mref.index as usize];
                                right[right_mref.index as usize] = tmp;

                                self.memory
                                    .insert(left_mref.name.clone(), MemoryContainer::Bit(left));
                                self.memory
                                    .insert(right_mref.name.clone(), MemoryContainer::Bit(right));
                            }
                            (
                                MemoryContainer::Integer(mut left),
                                MemoryContainer::Integer(mut right),
                            ) => {
                                let tmp = left[left_mref.index as usize];
                                left[left_mref.index as usize] = right[right_mref.index as usize];
                                right[right_mref.index as usize] = tmp;

                                self.memory
                                    .insert(left_mref.name.clone(), MemoryContainer::Integer(left));
                                self.memory.insert(
                                    right_mref.name.clone(),
                                    MemoryContainer::Integer(right),
                                );
                            }
                            (MemoryContainer::Real(mut left), MemoryContainer::Real(mut right)) => {
                                let tmp = left[left_mref.index as usize];
                                left[left_mref.index as usize] = right[right_mref.index as usize];
                                right[right_mref.index as usize] = tmp;

                                self.memory
                                    .insert(left_mref.name.clone(), MemoryContainer::Real(left));
                                self.memory
                                    .insert(right_mref.name.clone(), MemoryContainer::Real(right));
                            }
                            (left, right) => {
                                return Err(VMError::IncompatibleExchange {
                                    left: left.type_of(),
                                    right: right.type_of(),
                                })
                            }
                        }
                    }
                }
                (left, right) => {
                    return Err(VMError::InvalidExchange {
                        left: left.to_string(),
                        right: right.to_string(),
                    })
                }
            },

            Instruction::Load {
                destination,
                source,
                offset,
            } => {
                let dest_memory = self.memory.remove(&destination.name).ok_or(
                    VMError::UndefinedMemoryRegion {
                        name: destination.name.clone(),
                    },
                )?;
                let source_memory =
                    self.memory
                        .get(source)
                        .ok_or(VMError::UndefinedMemoryRegion {
                            name: source.clone(),
                        })?;
                let offset_memory =
                    self.memory
                        .get(&offset.name)
                        .ok_or(VMError::UndefinedMemoryRegion {
                            name: offset.name.clone(),
                        })?;

                match (dest_memory, source_memory, offset_memory) {
                    (
                        MemoryContainer::Bit(mut dest_bits),
                        MemoryContainer::Bit(source_ints),
                        MemoryContainer::Integer(offset_ints),
                    ) => {
                        dest_bits[destination.index as usize] =
                            source_ints[offset_ints[offset.index as usize] as usize];

                        self.memory
                            .insert(destination.name.clone(), MemoryContainer::Bit(dest_bits));
                    }
                    (
                        MemoryContainer::Integer(mut dest_ints),
                        MemoryContainer::Integer(source_ints),
                        MemoryContainer::Integer(offset_ints),
                    ) => {
                        dest_ints[destination.index as usize] =
                            source_ints[offset_ints[offset.index as usize] as usize];

                        self.memory.insert(
                            destination.name.clone(),
                            MemoryContainer::Integer(dest_ints),
                        );
                    }
                    (
                        MemoryContainer::Real(mut dest_reals),
                        MemoryContainer::Real(source_reals),
                        MemoryContainer::Integer(offset_ints),
                    ) => {
                        dest_reals[destination.index as usize] =
                            source_reals[offset_ints[offset.index as usize] as usize];

                        self.memory
                            .insert(destination.name.clone(), MemoryContainer::Real(dest_reals));
                    }
                    (destination, source, _) => {
                        return Err(VMError::IncompatibleLoad {
                            destination: destination.type_of(),
                            ssource: source.type_of(),
                        })
                    }
                }
            }

            Instruction::Store {
                destination,
                offset,
                source: ssource,
            } => {
                let dest_memory =
                    self.memory
                        .remove(destination)
                        .ok_or(VMError::UndefinedMemoryRegion {
                            name: destination.clone(),
                        })?;
                let offset_memory =
                    self.memory
                        .get(&offset.name)
                        .ok_or(VMError::UndefinedMemoryRegion {
                            name: offset.name.clone(),
                        })?;
                let offset_ints = offset_memory.clone().integer()?;

                match (dest_memory, ssource) {
                    (MemoryContainer::Bit(mut dest_bits), ArithmeticOperand::LiteralInteger(i)) => {
                        let index = offset_ints[offset.index as usize] as usize;
                        dest_bits[index] = *i;

                        self.memory
                            .insert(destination.clone(), MemoryContainer::Bit(dest_bits));
                    }
                    (
                        MemoryContainer::Bit(mut dest_bits),
                        ArithmeticOperand::MemoryReference(mref),
                    ) => {
                        let source_memory =
                            self.memory
                                .get(&mref.name)
                                .ok_or(VMError::UndefinedMemoryRegion {
                                    name: mref.name.clone(),
                                })?;
                        match source_memory {
                            MemoryContainer::Bit(source_bits) => {
                                let index = offset_ints[offset.index as usize] as usize;
                                dest_bits[index] = source_bits[mref.index as usize];
                            }
                            sref => {
                                return Err(VMError::IncompatibleStore {
                                    ssource: sref.type_of(),
                                    destination: "BIT".to_string(),
                                })
                            }
                        }

                        self.memory
                            .insert(destination.clone(), MemoryContainer::Bit(dest_bits));
                    }
                    (
                        MemoryContainer::Integer(mut dest_ints),
                        ArithmeticOperand::LiteralInteger(i),
                    ) => {
                        let index = offset_ints[offset.index as usize] as usize;
                        dest_ints[index] = *i;

                        self.memory
                            .insert(destination.clone(), MemoryContainer::Integer(dest_ints));
                    }
                    (
                        MemoryContainer::Integer(mut dest_ints),
                        ArithmeticOperand::MemoryReference(mref),
                    ) => {
                        let source_memory =
                            self.memory
                                .get(&mref.name)
                                .ok_or(VMError::UndefinedMemoryRegion {
                                    name: mref.name.clone(),
                                })?;
                        let source_ints = source_memory.clone().integer()?;
                        let index = offset_ints[offset.index as usize] as usize;
                        dest_ints[index] = source_ints[mref.index as usize];
                        self.memory
                            .insert(destination.clone(), MemoryContainer::Integer(dest_ints));
                    }
                    (MemoryContainer::Real(mut dest_reals), ArithmeticOperand::LiteralReal(r)) => {
                        let index = offset_ints[offset.index as usize] as usize;
                        dest_reals[index] = *r;

                        self.memory
                            .insert(destination.clone(), MemoryContainer::Real(dest_reals));
                    }
                    (
                        MemoryContainer::Real(mut dest_reals),
                        ArithmeticOperand::MemoryReference(mref),
                    ) => {
                        let source_memory =
                            self.memory
                                .get(&mref.name)
                                .ok_or(VMError::UndefinedMemoryRegion {
                                    name: mref.name.clone(),
                                })?;
                        let source_reals = source_memory.clone().real()?;
                        let index = offset_ints[offset.index as usize] as usize;
                        dest_reals[index] = source_reals[mref.index as usize];
                        self.memory
                            .insert(destination.clone(), MemoryContainer::Real(dest_reals));
                    }
                    (dest, ArithmeticOperand::LiteralReal(_)) => {
                        return Err(VMError::IncompatibleStore {
                            ssource: "REAL".to_string(),
                            destination: dest.type_of(),
                        })
                    }
                    (dest, ArithmeticOperand::LiteralInteger(_)) => {
                        return Err(VMError::IncompatibleStore {
                            ssource: "INTEGER".to_string(),
                            destination: dest.type_of(),
                        })
                    }
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
    pub fn run(&mut self) -> Result<bool, VMError> {
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

#[allow(unused_must_use)]
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
            MemoryContainer::Integer(r) => {
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
            MemoryContainer::Integer(r) => {
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
