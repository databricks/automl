class AutoMLRuntimeError(Exception):
    """
    Base exception class for AutoML Runtime errors.
    """

    def __init__(self, message="", *args):
        self.message = message
        super().__init__(message, *args)


class InvalidArgumentError(AutoMLRuntimeError):
    """
    Invalid argument set by user.
    """
    pass
