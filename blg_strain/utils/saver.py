import numpy as np
import h5py
import os

class Saver:
    @classmethod
    def load(cls, filename):
        return cls.load_hdf5(filename)

    @classmethod
    def load_npz(cls, filename):
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


    def save(self, filename, compression=None):
        '''
        Saves data to a file. Use this function to wrap either `safe_hdf5` or
        `save_npz`. We currently wrap `safe_hdf5`.
        '''
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.save_hdf5(filename, compression=compression)


    def save_hdf5(self, filename, compression=None):
        '''
        Saves data to a compressed .h5 file

        filename: full path of destination file
        compression: (int) specified as kwarg compression_opts for
            h5py.Group.create_dataset
        '''
        self.obj_filenames = []

        with h5py.File(filename, 'w') as f:
            # Recursively search dictionary structure of the class
            def write_layer(group, obj):
                classname = '{0}.{1}'.format(obj.__class__.__module__,
                                             obj.__class__.__name__)
                group.attrs['class'] = classname
                for k, v in obj.__dict__.items():
                    # Find objects
                    if hasattr(v, '__dict__'): # objects have __dict__
                        # Some objects are common to multiple instances of a
                        # class (e.g. a series of FilledBands would have the
                        # same BandStructure). Be sure to save separately.
                        classes_to_skip = ['BandStructure', 'StrainedLattice']
                        obj_class = v.__class__.__name__
                        if obj_class in classes_to_skip:
                            # Get the filename from the class and store it
                            obj_filename = getattr(v, 'filename')
                            if obj_filename not in self.obj_filenames:
                                self.obj_filenames.append(obj_filename)
                            continue
                        # If it's a class we want to save, create a group for it
                        group2 = group.create_group(k)
                        write_layer(group2, v)
                    else: # arrays, scalars, or even functions
                        kwargs = {}
                        if compression:
                            if type(v) is np.ndarray:
                                kwargs.update(dict(compression='gzip',
                                                compression_opts=compression))
                        try:
                            group.create_dataset(k, data=v, **kwargs)
                        except Exception as e: # it's a string or a function?
                            try: # list of strings?
                                group.create_dataset(k, data=np.array(v,
                                                                dtype='S'))
                            except: # probably a function (e.g. a spline)
                                group.create_dataset(k, data=[], **kwargs)
            write_layer(f, self)


    def save_npz(self, filename):
        '''
        Saves data to a compressed .npz file

        filename: full path of destination file
        '''
        if filename[-4:] != '.npz':
            filename += '.npz'

        # walk through dictionary and pop all keys that are 'sl' or 'bs'
        # This is not as general as I would like
        def walk(d):
            for k, v in d.items():
                if hasattr(v, '__dict__'):
                    walk(v.__dict__)
            for key in ['sl', 'bs']:
                d.pop(key, None)
            return d

        np.savez_compressed(filename, **walk(self.__dict__))
