class BaseStrategy:
    def generate_signals(self, data):
        raise NotImplementedError("You must implement generate_signals()")