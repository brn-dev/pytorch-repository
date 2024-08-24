
def sample(self, batch_size: int, with_replacement: bool = False) -> BufferSamples:
    batch_indices = np.random.choice(self.size, batch_size, replace=with_replacement)
    return self._get_samples(batch_indices)