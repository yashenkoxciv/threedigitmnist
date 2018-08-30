import re
import inflect
import numpy as np
import matplotlib.pyplot as plt

p = inflect.engine()


class NumbersVocabulary:
    def __init__(self, n, start, end, unknown, absent):
        v = {}
        idx_c = 0
        for i in range(10**(n-1), 10**n):
            w = p.number_to_words(i, andword='')
            for t in re.split(' |-', w):
                if t not in v:
                    v[t] = idx_c
                    idx_c += 1
        v[start] = idx_c; idx_c += 1
        v[end] = idx_c; idx_c += 1
        v[unknown] = idx_c; idx_c += 1
        v[absent] = idx_c; idx_c += 1
        self.n = n
        self.v = v
        self.iv = {i: n for n, i in self.v.items()}
        self.start = start
        self.end = end
        self.unknown = unknown
        self.absent = absent
    
    def __len__(self):
        return len(self.v)
    
    def number_to_sequence(self, n, maxlen):
        w = p.number_to_words(n, andword='')
        s = [self.v[self.start]]
        for t in re.split(' |-', w):
            if t in self.v:
                s.append(self.v[t])
            else:
                s.append(self.v[self.unknown])
        s.append(self.v[self.end])
        if len(s) > maxlen:
            s = s[:maxlen]
        else:
            s.extend([self.v[self.absent]]*(maxlen - len(s)))
        return s
    
    def sequence_to_onehot(self, s, pad_empty=True):
        onehot = np.zeros([len(s), len(self)])
        onehot[np.arange(len(s)), s] = 1
        if pad_empty:
            onehot[onehot[:, self.v[self.absent]] == 1] = 0
        return onehot
    
    def batch(self, x, y, batch_size):
        nonzero_idx = np.argwhere(y != 0)
        digits = np.hstack((
                np.random.choice(nonzero_idx.flatten(), [batch_size, 1]),
                np.random.choice(len(y), [batch_size, self.n-1])
        ))
        imgs, seq, oneh = [], [], []
        for i in range(batch_size):
            cimg = np.concatenate(x[digits[i, :]], axis=1)
            cnum = int(''.join([str(j) for j in y[digits[i, :]]]))
            cseq = self.number_to_sequence(cnum, self.n+3)
            cone = self.sequence_to_onehot(cseq)
            imgs.append(cimg)
            seq.append(cseq[:-1])
            oneh.append(cone[1:])
        return [
                np.array(imgs).reshape(-1, 28, 28*self.n, 1) / 256,
                np.array(seq)
        ], np.array(oneh)

def show_examples(nv, x_imgs, x_labels):
    x, y = nv.batch(x_imgs, x_labels, 1)
    s = ' '.join([nv.iv[t] for t in x[1][0]])
    p = ' '.join([nv.iv[t] for t in np.argmax(y, axis=2)[0]])
    print(x[1])
    print(y)
    plt.imshow(x[0][0], cmap='gray')
    plt.title('->'.join([s, p]))
    plt.show()

#nv = NumbersVocabulary(3, '<beg>', '<end>', '<unk>', '<abs>')

#from keras.datasets import mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x, y = nv.batch(x_train, y_train, 10, 3, 4)

#show_examples(nv, x_train, y_train)
