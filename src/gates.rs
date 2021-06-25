use itertools::Itertools;
use ndarray::{arr2, Array, Array2};
use num::complex::Complex64;

use std::convert::TryInto;

use crate::matrix::{C0, C1, I1};

#[derive(Default, Clone)]
pub struct QGate {
    matrix: Array2<Complex64>,
    qubits: Vec<u64>,
}

pub fn gate_matrix(name: String, params: Vec<f64>, qubits: Vec<u64>) -> QGate {
    match name.as_str() {
        "I" => i(qubits[0]),
        "X" => x(qubits[0]),
        "Y" => y(qubits[0]),
        "Z" => z(qubits[0]),
        "RX" => rx(params[0], qubits[0]),
        "RY" => ry(params[0], qubits[0]),
        "RZ" => rz(params[0], qubits[0]),
        "H" => h(qubits[0]),
        "CZ" => cz(qubits[0], qubits[1]),
        "CNOT" => cnot(qubits[0], qubits[1]),
        "SWAP" => swap(qubits[0], qubits[1]),
        "CCNOT" => ccnot(qubits[0], qubits[1], qubits[2]),
        _ => todo!(),
    }
}

// TODO Is it possible to define a SquareArray2 type (where dimensions are equal)
// TODO Likewise, an array type which is both square and each dimension is 2^n

impl QGate {
    fn lift_adjacent(&self, i: u64, n_qubits: u64) -> Array2<Complex64> {
        let gate_size = (self.matrix.shape()[0] as f64).log2() as u64;
        let bottom = Array::eye(2u64.pow(i.try_into().unwrap()).try_into().unwrap());
        let top =
            Array::eye(2u64.pow((n_qubits as u32) - (i as u32) - (gate_size as u32)) as usize);

        kron(&top, &kron(&self.matrix, &bottom))
    }

    pub fn lift(&self, n_qubits: u64) -> Array2<Complex64> {
        if self.qubits.len() > 2 {
            todo!("lifting is not yet supported for gates operating on more than two qubits")
        }

        let start = *self.qubits.iter().sorted().rev().collect_vec()[0];
        let (permutation, _) =
            juxtaposing_permutation_matrix(self.qubits[0], self.qubits[1], n_qubits);
        let permutation_dag = conj(permutation.clone().reversed_axes());
        let lifted = self.lift_adjacent(start, n_qubits);

        permutation_dag.dot(&lifted.dot(&permutation))
    }
}

fn conj(arr: Array2<Complex64>) -> Array2<Complex64> {
    let mut res = Array::zeros(arr.raw_dim());

    arr.indexed_iter()
        .for_each(|(d, v)| res[[d.0, d.1]] = v.conj());

    res
}

/// Creates a permutation matrix that moves the Hilbert space for qubit `i` into
/// a position adjacent to that of qubit `j`.
fn juxtaposing_permutation_matrix(j: u64, k: u64, n_qubits: u64) -> (Array2<Complex64>, Vec<u64>) {
    let mut permutation = Array::eye(2u64.pow(n_qubits as u32) as usize);
    let mut new_q_map = (0..n_qubits).collect();

    match j.cmp(&k) {
        std::cmp::Ordering::Equal => (permutation, new_q_map),
        std::cmp::Ordering::Less => {
            for i in j..k {
                permutation = swap(0, 1).lift_adjacent(i, n_qubits).dot(&permutation);
                new_q_map.swap(i as usize, (i + 1) as usize)
            }
            (permutation, new_q_map)
        }
        std::cmp::Ordering::Greater => {
            for i in j..k {
                permutation = swap(0, 1).lift_adjacent(i - 1, n_qubits).dot(&permutation);
                new_q_map.swap((i - 1) as usize, i as usize)
            }
            (permutation, new_q_map)
        }
    }
}

pub fn i(q: u64) -> QGate {
    QGate {
        matrix: arr2(&[[C1, C0], [C0, C1]]),
        qubits: [q].to_vec(),
    }
}

pub fn x(q: u64) -> QGate {
    QGate {
        matrix: arr2(&[[C0, C1], [C1, C0]]),
        qubits: [q].to_vec(),
    }
}

pub fn y(q: u64) -> QGate {
    QGate {
        matrix: arr2(&[[C0, -I1], [I1, C0]]),
        qubits: [q].to_vec(),
    }
}

pub fn z(q: u64) -> QGate {
    QGate {
        matrix: arr2(&[[C1, C0], [C0, -C1]]),
        qubits: [q].to_vec(),
    }
}

pub fn h(q: u64) -> QGate {
    QGate {
        matrix: arr2(&[
            [1.0 / 2.0f64.sqrt() + C0, 1.0 / 2.0f64.sqrt() + C0],
            [1.0 / 2.0f64.sqrt() + C0, -1.0 / 2.0f64.sqrt() + C0],
        ]),
        qubits: [q].to_vec(),
    }
}

pub fn rx(param: f64, q: u64) -> QGate {
    QGate {
        matrix: arr2(&[
            [(param / 2.0).cos() + C0, -I1 * (param / 2.0).sin()],
            [-I1 * (param / 2.0).sin(), (param / 2.0).cos() + C0],
        ]),
        qubits: [q].to_vec(),
    }
}

pub fn ry(param: f64, q: u64) -> QGate {
    QGate {
        matrix: arr2(&[
            [(param / 2.0).cos() + C0, -(param / 2.0).sin() + C0],
            [(param / 2.0).sin() + C0, (param / 2.0).cos() + C0],
        ]),
        qubits: [q].to_vec(),
    }
}

pub fn rz(param: f64, q: u64) -> QGate {
    QGate {
        matrix: arr2(&[
            [(-param / 2.0).cos() + I1 * (-param / 2.0).sin(), C0],
            [C0, (-param / 2.0).cos() + I1 * (param / 2.0).sin()],
        ]),
        qubits: [q].to_vec(),
    }
}

pub fn cz(control: u64, target: u64) -> QGate {
    QGate {
        matrix: arr2(&[
            [C1, C0, C0, C0],
            [C0, C1, C0, C0],
            [C0, C0, C1, C0],
            [C0, C0, C0, -C1],
        ]),
        qubits: [control, target].to_vec(),
    }
}

pub fn cnot(control: u64, target: u64) -> QGate {
    QGate {
        matrix: arr2(&[
            [C1, C0, C0, C0],
            [C0, C1, C0, C0],
            [C0, C0, C0, C1],
            [C0, C0, C1, C0],
        ]),
        qubits: [control, target].to_vec(),
    }
}

pub fn swap(q0: u64, q1: u64) -> QGate {
    QGate {
        matrix: arr2(&[
            [C1, C0, C0, C0],
            [C0, C0, C1, C0],
            [C0, C1, C0, C0],
            [C0, C0, C0, C1],
        ]),
        qubits: [q0, q1].to_vec(),
    }
}

pub fn ccnot(control0: u64, control1: u64, target: u64) -> QGate {
    QGate {
        matrix: arr2(&[
            [C1, C0, C0, C0, C0, C0, C0, C0],
            [C0, C1, C0, C0, C0, C0, C0, C0],
            [C0, C0, C1, C0, C0, C0, C0, C0],
            [C0, C0, C0, C1, C0, C0, C0, C0],
            [C0, C0, C0, C0, C1, C0, C0, C0],
            [C0, C0, C0, C0, C0, C1, C0, C0],
            [C0, C0, C0, C0, C0, C0, C0, C1],
            [C0, C0, C0, C0, C0, C0, C1, C0],
        ]),
        qubits: [control0, control1, target].to_vec(),
    }
}

/// Lifts the gate on the given target qubit into the full n_qubits Hilbert
/// space
// pub fn lift_oneq_gate_matrix(gate: QGate, n_qubits: u64, target_qubit: u64) -> QGate {
//     let mut lifted = gate;

//     for j in 0..n_qubits {
//         //println!("{:#}", lifted);
//         if j < target_qubit {
//             lifted = kron(&lifted, &i(0));
//         } else if j == target_qubit {
//             continue;
//         } else {
//             lifted = kron(&i(0), &lifted);
//         }
//     }

//     return lifted;
// }

/// Lifts the gate on the given target qubit into the full n_qubits Hilbert
/// space
// pub fn lift_twoq_gate_matrix(gate: QGate, n_qubits: u64, q0: u64, q1: u64) -> QGate {
//     if q0 < q1 && q1 - q0 == 1 {
//         return lift_adjacent_twoq_gate_matrix(gate, n_qubits, q0, q1);
//     } else if q1 < q0 && q0 - q1 == 1 {
//         return lift_adjacent_twoq_gate_matrix(gate, n_qubits, q1, q0);
//     }

//     let m = 2u64.pow(n_qubits.try_into().unwrap());
//     let mut right_swaps = ndarray::Array::eye(m.try_into().unwrap());
//     let mut left_swaps = ndarray::Array::eye(m.try_into().unwrap());
//     for i in q0..q1 - 1 {
//         let s = lift_adjacent_twoq_gate_matrix(swap(), n_qubits, i, i + 1);
//         right_swaps = right_swaps.dot(&s);
//         left_swaps = s.dot(&left_swaps);
//     }

//     return left_swaps.t().dot(&gate.dot(&right_swaps));
// }

// fn lift_adjacent_twoq_gate_matrix(gate: QGate, n_qubits: u64, q0: u64, q1: u64) -> QGate {
//     let mut lifted = gate;

//     if q0 > q1 {
//         let swap = lift_adjacent_twoq_gate_matrix(swap(), n_qubits, q1, q0);
//         let rev_lifted = lift_adjacent_twoq_gate_matrix(lifted, n_qubits, q1, q0);
//         return swap.t().dot(&rev_lifted.dot(&swap));
//     }

//     for j in 0..n_qubits {
//         if j < q0 {
//             lifted = kron(&lifted, &i())
//         } else if j > q1 {
//             lifted = kron(&i(), &lifted)
//         }
//     }

//     return lifted;
// }

#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use pretty_assertions::assert_eq;

    use crate::matrix::{C0, C1};

    #[test]
    fn lift_1q() {
        let gate = super::x(0);
        assert_eq!(
            gate.lift_adjacent(0, 2),
            arr2(&[
                [C0, C1, C0, C0],
                [C1, C0, C0, C0],
                [C0, C0, C0, C1],
                [C0, C0, C1, C0],
            ])
        );

        let gate = super::x(1);
        assert_eq!(
            gate.lift_adjacent(1, 2),
            arr2(&[
                [C0, C0, C1, C0],
                [C0, C0, C0, C1],
                [C1, C0, C0, C0],
                [C0, C1, C0, C0],
            ])
        );
    }

    #[test]
    fn test_adjacent_permutation_matrix() {
        // TODO
    }

    #[test]
    fn lift_2q() {
        let gate = super::cnot(0, 1);
        println!("{:#}", gate.lift(2));
    }
}

/// courtesy of https://github.com/rust-ndarray/ndarray/issues/652
// pub fn kron<T>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
// where
//     T: LinalgScalar,
// {
//     let dima = a.shape()[0];
//     let dimb = b.shape()[0];
//     let dimout = dima * dimb;
//     let mut out = Array2::zeros((dimout, dimout));
//     for (mut chunk, elem) in out.exact_chunks_mut((dimb, dimb)).into_iter().zip(a.iter()) {
//         chunk.assign(&(*elem * b));
//     }
//     out
// }

/// courtesy of https://github.com/rust-ndarray/ndarray/issues/652
pub fn kron(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let dima = a.shape()[0];
    let dimb = b.shape()[0];
    let dimout = dima * dimb;
    let mut out = Array2::zeros((dimout, dimout));
    for (mut chunk, elem) in out.exact_chunks_mut((dimb, dimb)).into_iter().zip(a.iter()) {
        chunk.assign(&(*elem * b));
    }
    out
}
