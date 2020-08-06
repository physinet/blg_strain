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
            value = data[attr]
            if value.ndim < 1:
                setattr(obj, attr, value.item())  # extract scalar
            else:
                setattr(obj, attr, value)

        return obj


    @classmethod
    def load_hdf5(cls, filename):
        '''
        Returns a Saver class object with parameters loaded from the .h5 file
        at location `filename`.
        '''

        with h5py.File(filename, 'r') as f:
            def read_layer(group):
                classname = group.attrs['class'].split('.')[-1]
                if classname == cls.__name__:
                    obj = cls()  # inizialize class object
                else:
                    try:
                        # Import all the possible classes
                        from ..lattice import StrainedLattice
                        from ..bands import Valley
                        from ..bands import BandStructure
                        from ..bands import FilledBands
                        obj = eval('{}()'.format(classname))  # instantiate
                    except:
                        raise Exception('Could not find {}'.format(classname))
                for k, v in group.items():
                    if isinstance(v, h5py.Group): # is a group
                        obj.__dict__[k] = read_layer(v)
                    else:
                        obj.__dict__[k] = v[()] # [()] gets values from Dataset
                return obj
            obj = read_layer(f)

        return obj


    def save_hdf5(self, filename):
        '''
        Saves data to a compressed .h5 file

        filename: full path of destination file
        '''
        with h5py.File(filename, 'w') as f:
            # Recursively search dictionary structure of the class
            def write_layer(group, obj):
                classname = '{0}.{1}'.format(obj.__class__.__module__,
                                             obj.__class__.__name__)
                group.attrs['class'] = classname
                for k, v in obj.__dict__.items():
                    if hasattr(v, '__dict__'): # classes have __dict__
                        group2 = group.create_group(k)
                        write_layer(group2, v)
                    else:
                        kwargs = {}
                        ''' Compression - didn't help '''
                        # if type(v) is np.ndarray:
                        #       # maximum compression for numpy arrays
                        #     pass
                        #     kwargs.update(dict(compression='gzip',
                        #                         compression_opts=1))
                        try:
                            group.create_dataset(k, data=v, **kwargs)
                        except: # in case we try to save a spline, for example
                            group.create_dataset(k, data=[], **kwargs)
            write_layer(f, self)


    def save(self, filename):
        '''
        Saves data to a compressed .npz file

        filename: full path of destination file
        '''
        if filename[-4:] != '.npz':
            filename += '.npz'

        np.savez_compressed(filename, **self.__dict__)
