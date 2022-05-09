class PosGenerator:

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __call__(self):
        n_batch = len(self.data)/self.batch_size
        batch_count = 0
        np.random.shuffle(self.data)
        while True:
            if batch_count == n_batch:
                batch_count = 0
                np.random.shuffle(self.data)
            batch = self.data[batch_count*self.batch_size:(batch_count+1)*self.batch_size]
            batch_count += 1
            yield tuple(batch)

class NegGenerator:

    def __init__(self, data, negative_factor, batch_size):
        self.data_len = data.shape[0]
        self.dim = data.shape[1]
        self.neg_size = negative_factor
        self.batch_size = batch_size
        self.data = data

    def __call__(self):
        while True:
            idx = np.random.choice(self.data_len, self.batch_size*self.neg_size)
            samples = self.data[idx].reshape(  self.batch_size,self.neg_size, self.dim)
            yield tuple(samples)