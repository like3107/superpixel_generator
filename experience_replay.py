import numpy as np
import random
import h5py

class BatchMemento:
    """
    Remember training instances and create (stochastically prioritized) replay batches
    """
    def __init__(self):
        self.memory = []
        self.epsilon = 0.1

    def __len__(self):
        return len(self.memory)

    def add_to_memory(self, batch_first, batch_second, add_info=None):
        """
        adds a batch with additional information (batch_second) to the
        memory.
        """
        if len(self) == 0:
            self.first_type = batch_first.dtype
            self.second_type = batch_second.dtype

        for i in range(len(batch_first)):
            sample = {"first":batch_first[i],
                                "second":batch_second[i],
                                "loss":None}
            if not add_info is None:
                sample.update(**add_info[i])
            self.memory.append(sample)

    def get_batch(self, batchsize):

        # return batch of max size if requested batch size is bigger than memory
        if (batchsize > len(self)):
            print "WARNING: requesting larger batch than memory, reducing output size"
            batchsize = len(self)


        # choose unknown batches first
        choices = [i for (i,m) in enumerate(self.memory) if m["loss"] is None][-batchsize:]

        # if there is space left in the batch add from memory
        residual = batchsize-len(choices)
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

    def get_evenheight_batch(self, batchsize):
        num_samples = len(self) 
        choices = np.random.choice(np.arange(len(self)),
                                       size=batchsize,
                                       replace=False,
                                       p=self.get_evenheight_priority()).tolist()

        return np.stack([self.memory[i]["first"] for i in choices])\
                                            .astype(self.first_type),\
               np.stack([self.memory[i]["second"] for i in choices])\
                                            .astype(self.second_type),\
               choices

    def update_loss(self, loss ,choices):
        for l,c in zip(loss, choices):
            self.memory[c]["loss"] = np.mean(l)

    def get_priority(self):
        # the epsilon offset ensures that all samples have a non zero chance
        # of being chosen. Samples with unknown loss are not chosen
        priority = np.array([0 if m["loss"] is None\
                            else m["loss"]+self.epsilon\
                            for m in self.memory],dtype=np.float32)

        priority /= np.sum(priority)
        return priority

    def get_evenheight_priority(self):
        heights = np.array([m["height"] for m in self.memory])
        maxheight = np.amax(heights)
        minheight = np.amin(heights)
        steps = np.linspace(minheight, maxheight, 10)
        digit = np.digitize(heights, steps)
        bincount = np.bincount(digit)
        priority = [1./bincount[s] for s in digit]
        priority /= np.sum(priority)
        return priority


    def count_new(self):
        return len([0 for m in self.memory if m["loss"] is None])

    def forget(self, p=0.1):
        random.shuffle(self.memory)
        self.memory = self.memory[:int(p*len(self))]

    def clear_memory(self):
        self.memory = []

    def save(self, file_name):
        with h5py.File(file_name, 'w') as out_h5: 
            out_h5.create_dataset("len",data=int(len(self)),dtype=int)
            out_h5.create_dataset("epsilon",data=float(self.epsilon),dtype=float)

            if len(self) > 0:
                keys = self.memory[0].keys()
                for k in keys:
                    out_h5.create_dataset("mem/"+k,data=np.array([np.nan if m[k] is None\
                            else m[k] for m in self.memory]),compression='gzip')

    def load(self, file_name):
        with h5py.File(file_name, 'r') as in_h5: 
            self.epsilon = in_h5['epsilon'].value
            if in_h5["len"].value > 0:
                mkeys = in_h5["mem"].keys()
                for i in range(in_h5["len"].value):
                    sample = {}
                    for k in mkeys:
                        print k,i
                        if k == "loss" and in_h5["mem/"+k][i] == np.nan:
                            sample[k] = None
                        else:
                            sample[k] = in_h5["mem/"+k][i]
                    self.memory.append(sample)
            else:
                self.memory = []
                self.priority = []

def stack_batch(b1, b2):
    return np.stack((b1,b2)).swapaxes(0,1)

def flatten_stack(batch):
    return np.concatenate((batch[:,0],batch[:,1]),axis=0)
    

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
