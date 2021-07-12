pub mod standard;

use ndarray::{Array, Array2};
use num::complex::Complex64;

use std::convert::TryInto;

use crate::gates::standard::{ccnot, cnot, cz, h, i, rx, ry, rz, swap, x, z};

#[derive(Default, Clone)]
pub struct QGate {
    matrix: Array2<Complex64>,
    qubits: Vec<u64>,
}

pub fn gate_matrix(name: String, params: Vec<f64>, qubits: Vec<u64>) -> QGate {
    match name.as_str() {
        "I" => i(qubits[0]),
        "X" => x(qubits[0]),
        "Y" => x(qubits[0]),
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
        match self.qubits.len().cmp(&2) {
            std::cmp::Ordering::Less => self.lift_adjacent(self.qubits[0], n_qubits),
            std::cmp::Ordering::Equal => {
                let (permutation, _, start) =
                    juxtaposing_permutation_matrix(self.qubits[0], self.qubits[1], n_qubits);
                let permutation_dag = conj(permutation.clone().reversed_axes());
                let lifted = self.lift_adjacent(start, n_qubits);

                permutation_dag.dot(&lifted.dot(&permutation))
            }
            std::cmp::Ordering::Greater => {
                todo!("lifting is not yet supported for gates operating on more than two qubits")
            }
        }
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
fn juxtaposing_permutation_matrix(
    j: u64,
    k: u64,
    n_qubits: u64,
) -> (Array2<Complex64>, Vec<u64>, u64) {
    let mut permutation = Array::eye(2u64.pow(n_qubits as u32) as usize);
    let mut new_q_map = (0..n_qubits).collect();

    match j.cmp(&k) {
        std::cmp::Ordering::Equal => (permutation, new_q_map, j),
        std::cmp::Ordering::Less => {
            for i in j..k {
                permutation = swap(0, 1).lift_adjacent(i, n_qubits).dot(&permutation);
                new_q_map.swap(i as usize, (i + 1) as usize)
            }
            (permutation, new_q_map, k - 1)
        }
        std::cmp::Ordering::Greater => {
            for i in (k + 1..j).rev() {
                permutation = swap(0, 1).lift_adjacent(i - 1, n_qubits).dot(&permutation);
                new_q_map.swap((i - 1) as usize, i as usize)
            }
            (permutation, new_q_map, j - 1)
        }
    }
}

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
        assert_eq!(
            gate.lift(2),
            arr2(&[
                [C1, C0, C0, C0],
                [C0, C0, C0, C1],
                [C0, C0, C1, C0],
                [C0, C1, C0, C0],
            ])
        );

        let gate = super::cnot(1, 0);
        assert_eq!(
            gate.lift(2),
            arr2(&[
                [C1, C0, C0, C0],
                [C0, C1, C0, C0],
                [C0, C0, C0, C1],
                [C0, C0, C1, C0],
            ])
        );

        let gate = super::cnot(0, 2);
        assert_eq!(
            gate.lift(3),
            arr2(&[
                [C1, C0, C0, C0, C0, C0, C0, C0],
                [C0, C0, C0, C0, C0, C1, C0, C0],
                [C0, C0, C1, C0, C0, C0, C0, C0],
                [C0, C0, C0, C0, C0, C0, C0, C1],
                [C0, C0, C0, C0, C1, C0, C0, C0],
                [C0, C1, C0, C0, C0, C0, C0, C0],
                [C0, C0, C0, C0, C0, C0, C1, C0],
                [C0, C0, C0, C1, C0, C0, C0, C0],
            ])
        );

        let gate = super::cnot(2, 0);
        assert_eq!(
            gate.lift(3),
            arr2(&[
                [C1, C0, C0, C0, C0, C0, C0, C0],
                [C0, C1, C0, C0, C0, C0, C0, C0],
                [C0, C0, C1, C0, C0, C0, C0, C0],
                [C0, C0, C0, C1, C0, C0, C0, C0],
                [C0, C0, C0, C0, C0, C1, C0, C0],
                [C0, C0, C0, C0, C1, C0, C0, C0],
                [C0, C0, C0, C0, C0, C0, C0, C1],
                [C0, C0, C0, C0, C0, C0, C1, C0],
            ])
        );
    }
}
