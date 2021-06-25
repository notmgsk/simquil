use std::{convert::TryInto, fmt};

use num::{complex::Complex64, pow};

use crate::{
    gates::QGate,
    matrix::{C0, C1},
};

//pub type Wavefunction =

#[derive(PartialEq)]
pub struct Wavefunction {
    wfn: ndarray::Array1<Complex64>,
    n_qubits: u64,
}

impl Wavefunction {
    pub fn new(n_qubits: u64) -> Self {
        Wavefunction::ground_state_wavefunction(n_qubits)
    }

    /// Create a ground state wavefunction |00..00> for n qubits,
    pub fn ground_state_wavefunction(n: u64) -> Self {
        let mut wfn = ndarray::Array::zeros(pow(2, n.try_into().unwrap()));
        wfn[0] = C1;

        Wavefunction { wfn, n_qubits: n }
    }

    fn probability(amplitude: Complex64) -> f64 {
        amplitude.re * amplitude.re + amplitude.im * amplitude.im
    }

    /// Compute the probability that the qubit is in an excited state.
    pub fn excited_state_probability(&self, qubit: &u64) -> f64 {
        // |0>  |1>  |2>  |3>
        // |00> |01> |10> |11>
        //  a%   b%   c%   d%
        //
        // excited prob of q0 = b% + d%

        let mut cum_prob = 0f64;

        for i in 0..self.wfn.len() {
            if (i as u64) & 2u64.pow((*qubit).try_into().unwrap()) == 1 {
                cum_prob += Wavefunction::probability(self.wfn[i])
            }
        }

        cum_prob
    }

    /// Collapse the state of the wavefunction due to the measurement of a
    /// qubit.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The measured qubit
    /// * `excited_probability` - The probability that `qubit` is in an excited
    ///   state in the wavefunction
    /// * `measured` - The result of the measurement (`0` or `1`)
    pub fn collapse_wavefunction(&mut self, qubit: &u64, excited_probability: f64, measured: u64) {
        let normalizer = if measured == 1 {
            1.0 / excited_probability.sqrt()
        } else {
            1.0 / (1.0 - excited_probability).sqrt()
        };

        for i in 0..self.wfn.len() {
            if 1 - measured == ((i as u64) >> qubit) & 1 {
                self.wfn[i] = C0
            } else {
                self.wfn[i] = normalizer * self.wfn[i]
            }
        }
    }

    /// Apply a gate matrix to the wavefunction
    pub fn apply(&mut self, gate: &QGate) {
        self.wfn = gate.lift(self.n_qubits).dot(&self.wfn)
    }
}

impl fmt::Debug for Wavefunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(self.wfn.iter().enumerate().for_each(|(i, amplitude)| {
            writeln!(
                f,
                "|{0:01$b}>: {2:.6}, {3:.0}%",
                i,
                self.n_qubits.try_into().unwrap(),
                amplitude,
                (Wavefunction::probability(*amplitude) * 100.0)
            )
            .ok();
        }))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::Wavefunction;
    use crate::{
        gates,
        matrix::{C0, C1},
    };

    #[test]
    fn apply_1q_gates() {
        let mut wfn = Wavefunction::ground_state_wavefunction(1);
        let gate = gates::i(0);
        wfn.apply(&gate);
        assert_eq!(wfn, Wavefunction::ground_state_wavefunction(1));

        let mut wfn = Wavefunction::ground_state_wavefunction(1);
        let gate = gates::x(0);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C0, C1]));

        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let gate = gates::x(0);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C0, C1, C0, C0]));

        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let gate = gates::x(0);
        wfn.apply(&gate);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C1, C0, C0, C0]));

        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let gate = gates::x(1);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C0, C0, C1, C0]));

        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let gate0 = gates::x(0);
        let gate1 = gates::x(1);
        wfn.apply(&gate0);
        wfn.apply(&gate1);
        assert_eq!(wfn.wfn, arr1(&[C0, C0, C0, C1]));

        let mut wfn = Wavefunction::ground_state_wavefunction(3);
        let gate = gates::x(2);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C0, C0, C0, C0, C1, C0, C0, C0]));

        let mut wfn = Wavefunction::ground_state_wavefunction(3);
        let gate0 = gates::x(0);
        let gate1 = gates::x(1);
        let gate2 = gates::x(2);
        wfn.apply(&gate0);
        wfn.apply(&gate1);
        wfn.apply(&gate2);
        assert_eq!(wfn.wfn, arr1(&[C0, C0, C0, C0, C0, C0, C0, C1]));
    }

    #[test]
    fn apply_adjacent_2q_gates() {
        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let gate = gates::cnot(0, 1);
        wfn.apply(&gate);
        assert_eq!(wfn.wfn, arr1(&[C1, C0, C0, C0]));

        let mut wfn = Wavefunction::ground_state_wavefunction(2);
        let x = gates::x(0);
        let cnot = gates::cnot(0, 1);
        wfn.apply(&x);
        wfn.apply(&cnot);
        assert_eq!(wfn.wfn, arr1(&[C0, C0, C0, C1]));
    }
}
