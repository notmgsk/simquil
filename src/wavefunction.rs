use std::fmt;

use ndarray::Array;
use num::{complex::Complex64, pow};
use thiserror::Error;

use crate::{
    gates::QGate,
    matrix::{C0, C1},
};

#[derive(Error, Debug)]
pub enum WavefunctionError {}

#[derive(PartialEq, Default)]
pub struct Wavefunction {
    pub wfn: ndarray::Array1<Complex64>,
    pub n_qubits: usize,
}

impl Wavefunction {
    pub fn new(n_qubits: usize) -> Self {
        Wavefunction::ground_state_wavefunction(n_qubits)
    }

    /// Create a ground state wavefunction |00..00> for n qubits,
    pub fn ground_state_wavefunction(n: usize) -> Self {
        let mut wfn = ndarray::Array::zeros(pow(2, n));
        wfn[0] = C1;

        Wavefunction { wfn, n_qubits: n }
    }

    fn probability(amplitude: Complex64) -> f64 {
        amplitude.re * amplitude.re + amplitude.im * amplitude.im
    }

    /// Compute the probability that the qubit is in the state excited `|1>`
    pub fn excited_state_probability(&self, qubit: usize) -> f64 {
        let mut cum_prob = 0f64;

        for i in 0..self.wfn.len() {
            if ((i as usize) >> qubit) & 1 == 1 {
                cum_prob += Wavefunction::probability(self.wfn[i])
            }
        }

        cum_prob
    }

    pub fn sample(&self, qubit: usize) -> usize {
        let r = rand::random::<f64>();
        let excited_prob = self.excited_state_probability(qubit);
        if r <= excited_prob {
            1
        } else {
            0
        }
    }

    pub fn measure(&mut self, qubit: usize) -> usize {
        let excited_prob = self.excited_state_probability(qubit);
        let collapsed_state = self.sample(qubit);
        self.collapse_wavefunction(qubit, excited_prob, collapsed_state);
        collapsed_state
    }

    /// Collapse the state of the wavefunction due to the measurement of a
    /// qubit.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The measured qubit
    /// * `excited_probability` - The probability that `qubit` is in an excited state
    /// * `measured` - The result of the measurement (`0` or `1`)
    pub fn collapse_wavefunction(
        &mut self,
        qubit: usize,
        excited_probability: f64,
        measured: usize,
    ) {
        let normalizer = if measured == 1 {
            1.0 / excited_probability.sqrt()
        } else {
            1.0 / (1.0 - excited_probability).sqrt()
        };

        for i in 0..self.wfn.len() {
            if 1 - measured == (i >> qubit) & 1 {
                self.wfn[i] = C0
            } else {
                self.wfn[i] = normalizer * self.wfn[i]
            }
        }
    }

    /// Apply a gate to the wavefunction
    pub fn apply(&mut self, gate: &QGate) {
        let subwfns_indices = get_indices_for_qubits(gate.qubits.clone(), self.n_qubits);
        for subwfn_indices in subwfns_indices {
            let subwfn_amplitudes = Array::from_vec(
                subwfn_indices
                    .iter()
                    .map(|i| self.wfn[*i])
                    .collect::<Vec<_>>(),
            );
            let subwfn_new_amplitudes = gate.matrix.dot(&subwfn_amplitudes);
            subwfn_indices
                .iter()
                .zip(subwfn_new_amplitudes)
                .for_each(|(i, a)| self.wfn[*i] = a);
        }
    }
}

/// Generate all bit strings (as integers up to 2^n with each bitstring having the restriction that for each qubit index
/// in `qubits` the corresponding value in the bitstring is specified in `restrictions`.
///
/// For example, `bitstrings_with_restrictions(vec![2, 0], vec![1, 0], 3)` produces all 2^3 bitstrings (as integers)
/// where the third bit is always 1 and the first bit is always 0, i.e [0b100, 0b110].
fn bitstrings_with_restrictions(
    qubits: Vec<usize>,
    restrictions: Vec<usize>,
    n_qubits: usize,
) -> Vec<usize> {
    (0..(2u32.pow(n_qubits as u32)))
        .filter(|i| {
            qubits
                .iter()
                .zip(&restrictions)
                .all(|(q, v)| ((i >> *q) & 0b1) == (*v as u32))
        })
        .map(|i| i as usize)
        .collect()
}

/// Generate all bitstrings up to 2^n with each bitstring represented by a vector of bits.
fn bitstrings(n: usize) -> Vec<Vec<usize>> {
    (0..(2u32.pow(n as u32)))
        .map(|v| (0..n).rev().map(|i| ((v >> i) & 1) as usize).collect())
        .collect()
}

fn get_indices_for_qubits(qubits: Vec<usize>, n_qubits: usize) -> Vec<Vec<usize>> {
    let b = bitstrings(qubits.len())
        .iter()
        .map(|r| bitstrings_with_restrictions(qubits.clone(), r.clone(), n_qubits))
        .collect::<Vec<_>>();

    let mut results: Vec<Vec<usize>> = vec![];
    let rows = b.len();
    let cols = b[0].len();
    // Essentially a matrix transpose
    for j in 0..cols {
        let mut new_row = vec![];
        #[allow(clippy::needless_range_loop)]
        for i in 0..rows {
            new_row.push(b[i][j])
        }
        results.push(new_row);
    }

    results
}

impl fmt::Debug for Wavefunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.wfn.iter().enumerate().for_each(|(i, amplitude)| {
            let prob = Wavefunction::probability(*amplitude);
            if prob > 1e-10 {
                writeln!(
                    f,
                    "|{0:01$b}>: {2:.6}, {3:.0}%",
                    i,
                    self.n_qubits,
                    amplitude,
                    prob * 100.0,
                )
                .ok();
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use crate::{
        gates,
        matrix::{C0, C1},
    };

    use super::Wavefunction;

    #[test]
    fn apply_1q_gates() {
        let mut wfn = Wavefunction::ground_state_wavefunction(1);
        let gate = gates::standard::i(0);
        wfn.apply(&gate);
        assert_eq!(wfn, Wavefunction::ground_state_wavefunction(1));

        let mut wfn = Wavefunction::ground_state_wavefunction(1);
        let gate = gates::standard::x(0);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C0, C1]));

        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let gate = gates::standard::x(0);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C0, C1, C0, C0]));

        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let gate = gates::standard::x(0);
        wfn.apply(&gate);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C1, C0, C0, C0]));

        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let gate = gates::standard::x(1);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C0, C0, C1, C0]));

        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let gate0 = gates::standard::x(0);
        let gate1 = gates::standard::x(1);
        wfn.apply(&gate0);
        wfn.apply(&gate1);
        assert_eq!(wfn.wfn, arr1(&[C0, C0, C0, C1]));

        let mut wfn = Wavefunction::ground_state_wavefunction(3);
        let gate = gates::standard::x(2);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C0, C0, C0, C0, C1, C0, C0, C0]));

        let mut wfn = Wavefunction::ground_state_wavefunction(3);
        let gate0 = gates::standard::x(0);
        let gate1 = gates::standard::x(1);
        let gate2 = gates::standard::x(2);
        wfn.apply(&gate0);
        wfn.apply(&gate1);
        wfn.apply(&gate2);
        assert_eq!(wfn.wfn, arr1(&[C0, C0, C0, C0, C0, C0, C0, C1]));
    }

    #[test]
    fn apply_adjacent_2q_gates() {
        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let gate = gates::standard::cnot(0, 1);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C1, C0, C0, C0]));

        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let x = gates::standard::x(0);
        let cnot = gates::standard::cnot(0, 1);
        wfn.apply(&x);
        wfn.apply(&cnot);
        assert_eq!(wfn.wfn, arr1(&[C0, C0, C0, C1]));
    }
}
