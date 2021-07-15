use thiserror::Error;
use crate::matrix::InstructionMatrixError;

#[derive(Error, Debug)]
pub enum VMError {
    /// Represents a failure due to providing an operand whose type is incompatible with the instruction
    #[error("Invalid type for instruction operand {name}: expected {expected} but found {actual}")]
    InvalidType {
        name: String,
        expected: String,
        actual: String,
    },

    /// Represents a failure due to a mismatch between the expected type of a memory region and its actual type
    #[error("Unexpected type for memory region: expected {expected} but found {actual}")]
    UnexpectedMemoryRegionType { expected: String, actual: String },

    /// Represents a failure due to a mismatch between the expected type of a literal value and its actual type
    #[error("Unexpected literal type: expected {expected} but found {actual}")]
    UnexpectedLiteralType { expected: String, actual: String },

    /// Represents a failure due to referencing an undefined memory region
    #[error("Undefined memory region {name:?}")]
    UndefinedMemoryRegion { name: String },

    /// Represents a failure due to using a qubit variable where a qubit index was expected
    #[error("Unresolved qubit variable {name:?}")]
    UnresolvedQubitVariable { name: String },

    /// Represents a failure due to JUMPing to an undefined target
    #[error("Undefined JUMP target {name:?}")]
    UndefinedJumpTarget { name: String },

    /// Represents a failure due to finding a duplicate LABEL
    #[error("Duplicate LABEL {name:?}")]
    DuplicateLabel { name: String },

    #[error("Invalid destination for MOVE: expected a memory region but found {actual}")]
    InvalidMoveDestination { actual: String },

    #[error("Incompatible destination for EXCHANGE: due to source type, expected {right} but found {left}")]
    IncompatibleExchange { left: String, right: String },

    #[error("Invalid EXCHANGE: operands must be memory regions but found {left} and {right}")]
    InvalidExchange { left: String, right: String },

    #[error("Incompatible destination for MOVE: due to source type, expected {ssource} but found {destination}")]
    IncompatibleLoad {
        ssource: String,
        destination: String,
    },

    #[error("Incompatible destination for STORE: due to source type, expected {ssource} but found {destination}")]
    IncompatibleStore {
        ssource: String,
        destination: String,
    },

    #[error("Error when creating matrix")]
    MatrixError(#[from] InstructionMatrixError),
}

// TODO VMError is nice but it doesn't show which instruction caused the error. Define a VMRunError
// which wraps a VMError and includes the erroneous instruction.
