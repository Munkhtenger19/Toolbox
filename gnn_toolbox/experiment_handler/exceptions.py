class GNNToolboxError(Exception):
    """Base class for all exceptions raised by the GNN toolbox."""
    pass

class ModelError(GNNToolboxError):
    """Exception raised when there is an error related to GNN model."""
    pass

class ModelCreationError(GNNToolboxError):
    """Exception raised when there is an error related to GNN model."""
    pass

class ModelTrainingError(GNNToolboxError):
    """Exception raised when there is an error related to GNN model."""
    pass

class DatasetCreationError(GNNToolboxError):
    """Exception raised when there is an error related to data handling."""
    pass

class DataPreparationError(GNNToolboxError):
    """Exception raised when there is an error related to data handling."""
    pass

class AttackError(GNNToolboxError):
    """Exception raised when there is an error related to adversarial attacks."""
    pass

class GlobalAttackError(GNNToolboxError):
    """Exception raised when there is an error related to adversarial attacks."""
    pass

class GlobalAttackCreationError(GNNToolboxError):
    """Exception raised when there is an error related to adversarial attacks."""
    pass

class LocalAttackError(GNNToolboxError):
    """Exception raised when there is an error related to adversarial attacks."""
    pass

class LocalAttackCreationError(GNNToolboxError):
    """Exception raised when there is an error related to adversarial attacks."""
    pass