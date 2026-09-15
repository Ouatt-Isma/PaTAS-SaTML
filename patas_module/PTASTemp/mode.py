from enum import Enum
class Mode(Enum):
        """ Four modes
        INFERENCE
        TRAINING 
        TRAINING_FEEDFORWARD
        TRAINING_BACKPROPAGATION 
        """
        INFERENCE = 1
        TRAINING = 4
        TRAINING_FEEDFORWARD = 2
        TRAINING_BACKPROPAGATION = 3
        END=-1
        def __str__(self) -> str:
            return self.name