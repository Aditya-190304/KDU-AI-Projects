"""Custom exceptions used by the pipeline."""


class MedicalExtractionError(Exception):
    """Base exception for extraction failures."""


class UnsupportedFileTypeError(MedicalExtractionError):
    """Raised when the input file is not supported."""


class CorruptedDocumentError(MedicalExtractionError):
    """Raised when a PDF cannot be opened or parsed."""


class MissingDependencyError(MedicalExtractionError):
    """Raised when an optional dependency is required for a feature."""
