class History:
    def __init__(self):
        self.history = []

    def append(self, loss, accuracy):
        self.history.append((loss, accuracy))

    