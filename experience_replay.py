import numpy as np
import random
import h5py

class BatcherBatcherBatcher:
    """
    Remember training instances and create (stochastically prioritized) replay batches
    """
    def __init__(self, scale_height_factor=None, max_mem_size=20000, pl=40, warmstart=1000):
        self.first = np.empty((max_mem_size, 4, pl, pl),dtype='float32')
        self.second = np.empty((max_mem_size, 4, 1, 1),dtype='float32')
        self.length = 0
        self.max_mem_size = max_mem_size
        # determined by histogram of cliped height map
        self.height_histo = np.array([0.01946579,
                                      0.16362278,
                                      0.15049561,
                                      0.12307298,
                                      0.13038894,
                                      0.09960873,
                                      0.09685287,
                                      0.08274214,
                                      0.0802579 ,
                                      0.05349225])
        self.scale_height_factor = scale_height_factor
        self.warmstart = warmstart
        assert(self.warmstart < self.max_mem_size)

    def __len__(self):
        return self.length

    def add_to_memory(self, batch_first, batch_second):
        """
        adds a batch with additional information (batch_second) to the
        memory.
        """
        bs = batch_first.shape[0]
        if self.length < self.warmstart:
            self.first[self.length:self.length+bs] = batch_first
            self.second[self.length:self.length+bs] = batch_second
            self.length += bs
        elif self.length == self.max_mem_size:
            # memory is full, replace based on height
            steps = np.linspace(0, 20*self.scale_height_factor, 2*self.scale_height_factor)
            digit = np.digitize(batch_second.mean(axis=1)[:,0,0], steps) -1
            probs = self.height_histo[digit]
            accepted = (probs > np.random.random(probs.shape))
            chosen_bs = np.sum(accepted)
            chosen_m = np.random.choice(np.arange(self.max_mem_size),
                                         size=chosen_bs,
                                         replace=False)
            self.first[chosen_m] = batch_first[accepted]
            self.second[chosen_m] = batch_second[accepted]
            self.length += chosen_bs
        elif 0 <= self.max_mem_size - (self.length + bs):
            # there is enough space to fit the whole batch
            # --> add subset of batch
            steps = np.linspace(0, 20*self.scale_height_factor, 10)
            digit = np.digitize(batch_second.mean(axis=1)[:,0,0], steps) -1
            probs = self.height_histo[digit]
            accepted = (probs > np.random.random(probs.shape))
            chosen_bs = np.sum(accepted)
            self.first[self.length:self.length+chosen_bs] = batch_first[accepted]
            self.second[self.length:self.length+chosen_bs] = batch_second[accepted]
            self.length += chosen_bs
        else:
            # the whole batch does not fit anymore(fill up memory)
            left_over = self.max_mem_size - self.length
            self.first[-self.length:] = batch_first[np.random.choice(np.arange(bs),
                                        size=left_over, replace=False)]
            self.length += left_over

    def get_batch(self, batchsize):
        # return batch of max size if requested batch size is bigger than memory
        if (batchsize > len(self)):
            print "WARNING: requesting larger batch than memory, reducing output size"
            batchsize = len(self)

        choices = np.random.choice(np.arange(len(self)),
                                           size=batchsize,
                                           replace=False)
        return self.first[choices], self.second[choices]


    def save(self, file_name):
        with h5py.File(file_name, 'w') as out_h5: 
            out_h5.create_dataset("length",data=int(len(self)),dtype=int)
            if not self.scale_height_factor is None:
                out_h5.create_dataset("scale_height_factor",data=float(self.scale_height_factor),dtype=float)
                out_h5.create_dataset("mem/second",data=self.second/float(self.scale_height_factor),dtype='float32',compression='gzip')
            else:
                out_h5.create_dataset("scale_height_factor",data=1.,dtype=float)
                out_h5.create_dataset("mem/second",data=self.second,dtype='float32',compression='gzip')
            out_h5.create_dataset("mem/first",data=self.first,dtype='float32')

    def load(self, file_name):
        with h5py.File(file_name, 'r') as in_h5: 
            self.length = in_h5["length"].value
            self.first = in_h5["mem/first"].value
            self.second = in_h5["mem/second"].value
            self.max_mem_size = in_h5["mem/first"].shape[0]
            if not self.scale_height_factor is None:
                self.second *= self.scale_height_factor

def stack_batch(b1, b2):
    return np.stack((b1,b2)).swapaxes(0,1)

def flatten_stack(batch):
    return np.concatenate((batch[:,0],batch[:,1]),axis=0)
    

class BatchMemento:
    """
    Remember training instances and create (stochastically prioritized) replay batches
    """
    def __init__(self, scale_height_factor=None):
        self.memory = []
        self.epsilon = 0.1
        self.scale_height_factor = scale_height_factor

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
                    if k == "height" and not self.scale_height_factor in None:
                        out_h5.create_dataset("mem/"+k,data=np.array([np.nan if m[k] is None\
                            else m[k]/self.scale_height_factor for m in self.memory]),compression='gzip')
                    else:
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
                        if k == "loss" and in_h5["mem/"+k][i] == np.nan:
                            sample[k] = None
                        else:
                            sample[k] = in_h5["mem/"+k][i]
                            if k == "height" and not self.scale_height_factor in None:
                                sample[k] *= self.scale_height_factor
                    self.memory.append(sample)
                    
                self.first_type = self.memory[0]["first"].dtype
                self.second_type = self.memory[0]["second"].dtype
            else:
                self.memory = []
                self.priority = []

def stack_batch(b1, b2):
    return np.stack((b1,b2)).swapaxes(0,1)

def flatten_stack(batch):
    return np.concatenate((batch[:,0],batch[:,1]),axis=0)
    

if __name__ == '__main__':
    # M = BatchMemento()

    # for i in range(30):
    #     batch = np.random.random((3,4,40,40)).astype(np.float32)
    #     gt = np.random.random((3,4,1,1)).astype(np.float32)

    #     M.add_to_memory(batch, gt)
    #     f,s,c = M.get_batch(10)
    #     print "ce",c
    #     print f.shape,s.shape,c
    #     M.update_loss(np.random.random((10)),c)
    M = BatcherBatcherBatcher(scale_height_factor=10, max_mem_size=31, warmstart=4)

    for i in range(30):
        batch = np.random.random((3,4,40,40)).astype(np.float32)
        gt = np.random.random((3,4,1,1)).astype(np.float32)*100

        M.add_to_memory(batch, gt)
        f,s = M.get_batch(20)
        print len(M)
        print f.shape,s.shape

    M.save('exp_test.h5')
    M2 = BatcherBatcherBatcher(scale_height_factor=2., max_mem_size=31, warmstart=4)
    M2.load('exp_test.h5')
    M2.save('exp_test2.h5')


    M3 = BatcherBatcherBatcher(scale_height_factor=100, max_mem_size=31, warmstart=4)
    M3.load('exp_test2.h5')
    M3.save('exp_test3.h5')
