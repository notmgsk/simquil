use crate::matrix::{C0, C1, I1};
use ndarray::arr2;
use std::f64::consts::FRAC_PI_4;
use thiserror::Error;

use super::QGate;

#[derive(Error, Debug)]
pub enum GateError {
    #[error("Unknown gate {0}")]
    UnknownGate(String),
    #[error("Gate matrix missing qubit")]
    GateMatrixMissingQubit,
    #[error("Gate matrix missing parameter")]
    GateMatrixMissingParameter,
}

macro_rules! define_gate {
    ($name:ident, [$($qubit:ident),+], $gate:expr) => {
        pub fn $name($($qubit: usize),*) -> QGate {
            QGate {
                matrix: arr2(&$gate),
                qubits: [$($qubit),*].to_vec(),
            }
        }
    };
    ($name:ident, [$($param:ident),+], [$($qubit:ident),+], $gate:expr) => {
        pub fn $name($($param: f64),*,$($qubit: usize),*) -> QGate {
            QGate {
                matrix: arr2(&$gate),
                qubits: [$($qubit),*].to_vec(),
            }
        }
    };
}

define_gate!(i, [q], [[C1, C0], [C0, C1]]);
define_gate!(x, [q], [[C0, C1], [C1, C0]]);
define_gate!(y, [q], [[C0, -I1], [I1, C0]]);
define_gate!(z, [q], [[C1, C0], [C0, -C1]]);
define_gate!(
    h,
    [q],
    [
        [1.0 / 2.0f64.sqrt() + C0, 1.0 / 2.0f64.sqrt() + C0],
        [1.0 / 2.0f64.sqrt() + C0, -1.0 / 2.0f64.sqrt() + C0],
    ]
);
define_gate!(s, [q], [[C1, C0], [C0, I1]]);
define_gate!(t, [q], [[C1, C0], [C0, FRAC_PI_4.cos() + C0]]);

define_gate!(
    rx,
    [theta],
    [q],
    [
        [(theta / 2.0).cos() + C0, -I1 * (theta / 2.0).sin()],
        [-I1 * (theta / 2.0).sin(), (theta / 2.0).cos() + C0],
    ]
);
define_gate!(
    ry,
    [theta],
    [q],
    [
        [(theta / 2.0).cos() + C0, -(theta / 2.0).sin() + C0],
        [(theta / 2.0).sin() + C0, (theta / 2.0).cos() + C0],
    ]
);
define_gate!(
    rz,
    [theta],
    [q],
    [
        [(-theta / 2.0).cos() + I1 * (-theta / 2.0).sin(), C0],
        [C0, (-theta / 2.0).cos() + I1 * (theta / 2.0).sin()],
    ]
);
define_gate!(
    phase,
    [theta],
    [q],
    [[C1, C0], [C0, theta.cos() + I1 * theta.sin()]]
);
define_gate!(
    cphase,
    [theta],
    [control, target],
    [
        [C1, C0, C0, C0],
        [C0, C1, C0, C0],
        [C0, C0, C1, C0],
        [C0, C0, C0, theta.cos() + I1 * theta.sin()],
    ]
);

define_gate!(
    cz,
    [control, target],
    [
        [C1, C0, C0, C0],
        [C0, C1, C0, C0],
        [C0, C0, C1, C0],
        [C0, C0, C0, -C1],
    ]
);
define_gate!(
    cnot,
    [control, target],
    [
        [C1, C0, C0, C0],
        [C0, C1, C0, C0],
        [C0, C0, C0, C1],
        [C0, C0, C1, C0],
    ]
);
define_gate!(
    swap,
    [q0, q1],
    [
        [C1, C0, C0, C0],
        [C0, C0, C1, C0],
        [C0, C1, C0, C0],
        [C0, C0, C0, C1],
    ]
);
define_gate!(
    iswap,
    [q0, q1],
    [
        [C1, C0, C0, C0],
        [C0, C0, I1, C0],
        [C0, I1, C0, C0],
        [C0, C0, C0, C1],
    ]
);
define_gate!(
    cswap,
    [control, q0, q1],
    [
        [C1, C0, C0, C0, C0, C0, C0, C0],
        [C0, C1, C0, C0, C0, C0, C0, C0],
        [C0, C0, C1, C0, C0, C0, C0, C0],
        [C0, C0, C0, C1, C0, C0, C0, C0],
        [C0, C0, C0, C0, C1, C0, C0, C0],
        [C0, C0, C0, C0, C0, C0, C1, C0],
        [C0, C0, C0, C0, C0, C1, C0, C0],
        [C0, C0, C0, C0, C0, C0, C0, C1],
    ]
);
define_gate!(
    ccnot,
    [control0, control1, target],
    [
        [C1, C0, C0, C0, C0, C0, C0, C0],
        [C0, C1, C0, C0, C0, C0, C0, C0],
        [C0, C0, C1, C0, C0, C0, C0, C0],
        [C0, C0, C0, C1, C0, C0, C0, C0],
        [C0, C0, C0, C0, C1, C0, C0, C0],
        [C0, C0, C0, C0, C0, C1, C0, C0],
        [C0, C0, C0, C0, C0, C0, C0, C1],
        [C0, C0, C0, C0, C0, C0, C1, C0],
    ]
);

pub fn gate_matrix(name: String, params: Vec<f64>, qubits: Vec<usize>) -> Result<QGate, GateError> {
    match name.as_str() {
        "I" => {
            let q = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(i(*q))
        }
        "X" => {
            let q = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(x(*q))
        }
        "Y" => {
            let q = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(y(*q))
        }
        "Z" => {
            let q = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(z(*q))
        }
        "H" => {
            let q = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(h(*q))
        }
        "T" => {
            let q = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(t(*q))
        }
        "S" => {
            let q = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(s(*q))
        }
        "RX" => {
            let q = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            let p = params.get(0).ok_or(GateError::GateMatrixMissingParameter)?;
            Ok(rx(*p, *q))
        }
        "RY" => {
            let q = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            let p = params.get(0).ok_or(GateError::GateMatrixMissingParameter)?;
            Ok(ry(*p, *q))
        }
        "RZ" => {
            let q = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            let p = params.get(0).ok_or(GateError::GateMatrixMissingParameter)?;
            Ok(rz(*p, *q))
        }
        "PHASE" => {
            let q = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            let p = params.get(0).ok_or(GateError::GateMatrixMissingParameter)?;
            Ok(phase(*p, *q))
        }
        "CZ" => {
            let control = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            let target = qubits.get(1).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(cz(*control, *target))
        }
        "CNOT" => {
            let control = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            let target = qubits.get(1).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(cnot(*control, *target))
        }
        "SWAP" => {
            let q0 = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            let q1 = qubits.get(1).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(cz(*q0, *q1))
        }
        "ISWAP" => {
            let q0 = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            let q1 = qubits.get(1).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(iswap(*q0, *q1))
        }
        "CSWAP" => {
            let control0 = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            let control1 = qubits.get(1).ok_or(GateError::GateMatrixMissingQubit)?;
            let target = qubits.get(2).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(cswap(*control0, *control1, *target))
        }
        "CCNOT" => {
            let control0 = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            let control1 = qubits.get(1).ok_or(GateError::GateMatrixMissingQubit)?;
            let target = qubits.get(1).ok_or(GateError::GateMatrixMissingQubit)?;
            Ok(ccnot(*control0, *control1, *target))
        }
        "CPHASE" => {
            let control = qubits.get(0).ok_or(GateError::GateMatrixMissingQubit)?;
            let target = qubits.get(1).ok_or(GateError::GateMatrixMissingQubit)?;
            let theta = params.get(0).ok_or(GateError::GateMatrixMissingParameter)?;
            Ok(cphase(*theta, *control, *target))
        }
        _ => Err(GateError::UnknownGate(name)),
    }
}
