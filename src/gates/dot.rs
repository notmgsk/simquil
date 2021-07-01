use ndarray::{arr2, linalg::Dot};

use crate::matrix::C1;

use super::ComplexMatrix;

#[cfg(feature = "blas")]
use cblas_sys as blas_sys;

impl Dot<ComplexMatrix> for ComplexMatrix {
    type Output = ComplexMatrix;
    fn dot(&self, b: &ComplexMatrix) -> ComplexMatrix {
        ComplexMatrix(arr2(&[[C1]]))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr2, linalg::Dot};

    use crate::{gates::ComplexMatrix, matrix::C1};

    #[test]
    fn a() {
        let x = ComplexMatrix(arr2(&[[C1]]));

        println!("{:#?}", x.dot(&x));
    }
}
