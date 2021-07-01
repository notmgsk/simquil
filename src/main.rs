use itertools::Itertools;
use quil::program::Program;
use std::{
    fs,
    io::{self, Read},
};
use structopt::StructOpt;
use vm::VM;

pub mod gates;
pub mod matrix;
pub mod vm;
pub mod wavefunction;

#[derive(StructOpt)]
struct Cli {
    quil_file: Option<String>,
}

fn run(cli: Cli) {
    let quil = match cli.quil_file {
        Some(path) => fs::read_to_string(path).expect("bad read"),
        None => {
            let mut stdin = io::stdin();
            let mut buf = String::new();
            stdin.read_to_string(&mut buf).expect("bad read from stdin");
            buf
        }
    };
    let program = Program::from_str(&quil).expect("bad program");

    let mut vm = VM::new(2, program);

    vm.run();
    println!("Wavefunction amplitudes:");
    println!();
    println!("{:?}", vm);

    println!("Classical memory:");
    println!();
    vm.memory
        .iter()
        .sorted_by_key(|x| x.0)
        .for_each(|(mref, data)| println!("{}[0..{}]:\t {:?}", mref, data.len(), data));
}

fn main() {
    run(Cli::from_args())
}
