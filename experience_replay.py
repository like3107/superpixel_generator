import numpy as np


class BatchMemento:
    """
    Remember training instances and create (stochastically prioritized) replay batches
    """
    def __init__(self):
        self.memory = []
        self.priority = []
        self.epsilon = 0.1

    def __len__(self):
        return len(self.memory)

    def add_to_memory(self, batch_first, batch_second):
        """
        adds a batch with additional information (batch_second) to the
        memory.
        """
        if len(self) == 0:
            self.first_type = batch_first.dtype
            self.second_type = batch_second.dtype

        for i in range(len(batch_first)):
            self.memory.append({"first":batch_first[i],
                                "second":batch_second[i],
                                "loss":None})

    def get_batch(self, batchsize):

        # return batch of max size if requested batch size is bigger than memory
        if (batchsize > len(self)):
            print "WARNING: requesting larger batch than memory, reducing output size"
            batchsize = len(self)


        # choose unknown batches first
        choices = [i for (i,m) in enumerate(self.memory) if m["loss"] is None][-batchsize:]

        # if there is space left in the batch add from memory
        residual = batchsize-len(choices)
        print "choices",choices
        if residual > 0:
            choices += np.random.choice(np.arange(len(self)),
                                           size=residual,
                                           replace=False,
                                           p=self.get_priority()).tolist()

        assert(len(choices)==batchsize)
        return np.stack([self.memory[i]["first"] for i in choices])\
                                            .astype(self.first_type),\
               np.stack([self.memory[i]["second"] for i in choices])\
                                            .astype(self.second_type),\
               choices

    def update_loss(self, loss ,choices):
        for l,c in zip(loss, choices):
            self.memory[c]["loss"] = l

    def get_priority(self):
        # the epsilon offset ensures that all samples have a non zero chance
        # of being chosen. Samples with unknown loss are not chosen
        priority = np.array([0 if m["loss"] is None\
                            else m["loss"]+self.epsilon\
                            for m in self.memory],dtype=np.float32)


        priority /= np.sum(priority)
        return priority

    def clear_memory(self):
        self.memory = []
        self.priority = []


if __name__ == '__main__':
    M = BatchMemento()

    for i in range(30):
        batch = np.random.random((3,4,40,40)).astype(np.float32)
        gt = np.random.random((3,4,1,1)).astype(np.float32)

        M.add_to_memory(batch, gt)
        f,s,c = M.get_batch(10)
        print "ce",c
        print f.shape,s.shape,c
        M.update_loss(np.random.random((10)),c)
