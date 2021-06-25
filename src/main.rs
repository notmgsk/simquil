use quil::program::Program;
use std::fs;
use structopt::StructOpt;
use vm::VM;

//use crate::gates::gate_matrix;

pub mod gates;
pub mod matrix;
pub mod vm;
pub mod wavefunction;

#[derive(StructOpt)]
struct Cli {
    quil_file: String,
}

fn main() {
    let args = Cli::from_args();
    let quil = fs::read_to_string(args.quil_file).expect("bad read");
    let program = Program::from_str(&quil).expect("bad program");

    println!("Input program:");
    println!("{}", program.to_string(true));

    let mut vm = VM::new(2, program);

    vm.run();
    println!("{:?}", vm);

    //let i = gate_matrix(String::from("I"), Vec::new());
    //let x = gate_matrix(String::from("X"), Vec::new());
    //println!("{:#}", gates::lift_gate_matrix(x, 2, 0));
}
