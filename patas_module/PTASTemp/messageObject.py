from PTASTemp.mode import Mode
class MessageObject:
    def __init__(self, _mode: Mode, _content: dict = None, _epoch: int = None , _batch: int = None, _layer: int =None):
        self.mode = _mode 
        self.content = _content
        self.epoch = _epoch 
        self.batch = _batch 
        self.layer = _layer
    def __str__(self) -> str:
        if (self.mode == Mode.INFERENCE):
            return f"mode: {self.mode} -- content: {self.content}"
        else:
            return f"mode: {self.mode} -- epoch:{self.epoch} -- batch:{self.batch} -- layer:{self.layer} -- content: {self.content}"
    
    def easy_print(self):
        if (self.mode == Mode.INFERENCE):
            return f"mode: {self.mode}"
        else:
            return f"mode: {self.mode} -- epoch:{self.epoch} -- batch:{self.batch} -- layer:{self.layer} "
    
    def __repr__(self) -> str:
        return self.__str__()