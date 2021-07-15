use ndarray::arr2;

use crate::matrix::{C0, C1, I1};

use super::QGate;

pub fn i(q: usize) -> QGate {
    QGate {
        matrix: arr2(&[[C1, C0], [C0, C1]]),
        qubits: [q].to_vec(),
    }
}

pub fn x(q: usize) -> QGate {
    QGate {
        matrix: arr2(&[[C0, C1], [C1, C0]]),
        qubits: [q].to_vec(),
    }
}

pub fn y(q: usize) -> QGate {
    QGate {
        matrix: arr2(&[[C0, -I1], [I1, C0]]),
        qubits: [q].to_vec(),
    }
}

pub fn z(q: usize) -> QGate {
    QGate {
        matrix: arr2(&[[C1, C0], [C0, -C1]]),
        qubits: [q].to_vec(),
    }
}

pub fn h(q: usize) -> QGate {
    QGate {
        matrix: arr2(&[
            [1.0 / 2.0f64.sqrt() + C0, 1.0 / 2.0f64.sqrt() + C0],
            [1.0 / 2.0f64.sqrt() + C0, -1.0 / 2.0f64.sqrt() + C0],
        ]),
        qubits: [q].to_vec(),
    }
}

pub fn rx(param: f64, q: usize) -> QGate {
    QGate {
        matrix: arr2(&[
            [(param / 2.0).cos() + C0, -I1 * (param / 2.0).sin()],
            [-I1 * (param / 2.0).sin(), (param / 2.0).cos() + C0],
        ]),
        qubits: [q].to_vec(),
    }
}

pub fn ry(param: f64, q: usize) -> QGate {
    QGate {
        matrix: arr2(&[
            [(param / 2.0).cos() + C0, -(param / 2.0).sin() + C0],
            [(param / 2.0).sin() + C0, (param / 2.0).cos() + C0],
        ]),
        qubits: [q].to_vec(),
    }
}

pub fn rz(param: f64, q: usize) -> QGate {
    QGate {
        matrix: arr2(&[
            [(-param / 2.0).cos() + I1 * (-param / 2.0).sin(), C0],
            [C0, (-param / 2.0).cos() + I1 * (param / 2.0).sin()],
        ]),
        qubits: [q].to_vec(),
    }
}

pub fn cz(control: usize, target: usize) -> QGate {
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

pub fn cnot(control: usize, target: usize) -> QGate {
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

pub fn swap(q0: usize, q1: usize) -> QGate {
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

pub fn ccnot(control0: usize, control1: usize, target: usize) -> QGate {
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
