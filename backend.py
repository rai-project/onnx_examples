class Backend:
    def __init__(self):
        pass

    def load(self, model):
        raise NotImplementedError("Backend:load")

    def forward_once(self, warmup=True):
        raise NotImplementedError("Backend:forward_once")

    def forward(self, warmup=True):
        raise NotImplementedError("Backend:forward")
