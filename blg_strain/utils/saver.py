import numpy as np
import h5py

class Saver:
    @classmethod
    def load(cls, filename):
        '''
        Returns a Saver class object with parameters loaded from the .npz file
        at location `filename`.
        '''

        with h5py.File(filename, 'r') as f:
            def read_layer(group):
                obj = cls()  # inizialize class object
                for k, v in group.items():
                    if isinstance(v, h5py.Group): # is a group
                        obj.__dict__[k] = read_layer(v)
                    else:
                        obj.__dict__[k] = v[()] # [()] gets values from h5py.Dataset
                return obj
            obj = read_layer(f)

        return obj


    def save(self, filename):
        '''
        Saves data to a compressed .h5 file

        filename: full path of destination file
        '''
        with h5py.File(filename, 'w') as f:
            # Recursively search dictionary structure of the class
            def write_layer(group, d):
                for k, v in d.items():
                    if hasattr(v, '__dict__'): # classes have __dict__
                        group2 = group.create_group(k)
                        write_layer(group2, v.__dict__)
                    else:
                        group.create_dataset(k, data=v)
            write_layer(f, self.__dict__)
