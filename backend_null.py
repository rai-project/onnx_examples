import backend


class BackendNull(backend.Backend):
    def __init__(self):
        super(BackendNull, self).__init__()

    def name(self):
        return "null"

    def version(self):
        return 0

    def load(self, model):
        pass

    def forward(self, img, warmup=True):
        return 0
