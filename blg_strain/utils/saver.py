import numpy as np
import h5py

class Saver:
    @classmethod
    def load(cls, filename):
        '''
        Returns a Saver class object with parameters loaded from the .npz file
        at location `filename`.
        '''
        obj = cls()  # inizialize class object

        data = np.load(filename, allow_pickle=True)

        # Set saved variables as attributes to the class object
        for attr in data.files:
            setattr(obj, attr, data[attr].item())

        return obj


    def save(self, filename):
        '''
        Saves data to a compressed .npz file

        filename: full path of destination file
        '''
        if filename[-4:] != '.npz':
            filename += '.npz'

        np.savez_compressed(filename, **self.__dict__)
