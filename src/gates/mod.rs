pub mod standard;

use ndarray::{Array, Array2};
use num::complex::Complex64;

use crate::gates::standard::swap;

#[derive(Default, Clone, Debug)]
pub struct QGate {
    pub matrix: Array2<Complex64>,
    pub qubits: Vec<usize>,
}

impl QGate {
    fn lift_adjacent(&self, i: usize, n_qubits: usize) -> Array2<Complex64> {
        let gate_size = self.qubits.len();
        let n = num::pow(2, i);
        let bottom = Array::eye(n);
        let top = Array::eye(2u64.pow((n_qubits - i - gate_size) as u32) as usize);

        kron(&top, &kron(&self.matrix, &bottom))
    }

    pub fn lift(&self, n_qubits: usize) -> Array2<Complex64> {
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
    j: usize,
    k: usize,
    n_qubits: usize,
) -> (Array2<Complex64>, Vec<usize>, usize) {
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

    use crate::gates::standard::{cnot, x};
    use crate::matrix::{C0, C1};

    #[test]
    fn lift_1q() {
        let gate = x(0);
        assert_eq!(
            gate.lift_adjacent(0, 2),
            arr2(&[
                [C0, C1, C0, C0],
                [C1, C0, C0, C0],
                [C0, C0, C0, C1],
                [C0, C0, C1, C0],
            ])
        );

        let gate = x(1);
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
        let gate = cnot(0, 1);
        assert_eq!(
            gate.lift(2),
            arr2(&[
                [C1, C0, C0, C0],
                [C0, C0, C0, C1],
                [C0, C0, C1, C0],
                [C0, C1, C0, C0],
            ])
        );

        let gate = cnot(1, 0);
        assert_eq!(
            gate.lift(2),
            arr2(&[
                [C1, C0, C0, C0],
                [C0, C1, C0, C0],
                [C0, C0, C0, C1],
                [C0, C0, C1, C0],
            ])
        );

        let gate = cnot(0, 2);
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

        let gate = cnot(2, 0);
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
