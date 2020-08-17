import numpy as np
import h5py
import os
import glob
import errno
import pickle

def load(path, parameters_only=False, load_summary=True):
    '''
    Returns values of Delta, EF, n, D, and alpha from one run of calculations.
    If a `summary.h5` file does not exist under the given path, this function
    will create a `Saver` object with the quantities extracted from the saved
    `FilledBands` files and save it to `summary.h5`. If this file does exist,
    this function instead loads directly from the summary file (to save time)
    unless `load_summary` is False.

    Parameters:
    - path: base directory associated with the run (e.g. `StrainedLattice_eps0.010_theta0.100_Run3`)
    - parameters_only: if True, only return Delta and EF (faster)
    - load_summary: if True, attempt to load from a `summary.h5` file rather
        than loading each `FilledBands`.

    Returns:
    - Deltas, EFs, [ns, Ds, alphas]
    '''

    if load_summary:
        filename = os.path.join(path, 'summary.h5')
        if os.path.exists(filename):
            s = Saver.load(filename)
            return s.Deltas, s.EFs, s.ns, s.Ds, s.alphas

    bs_paths = glob.glob(path + r'\BandStructure*.h5')  # Find all BandStructure files
    bs_paths.sort(key=os.path.getmtime)

    # All bs_paths have the same EF series; use the first bs_path
    fb_paths = glob.glob(bs_paths[0][:-3] + '\FilledBands*.h5')
    fb_paths.sort(key=os.path.getmtime)

    Deltas = np.empty(len(bs_paths))
    EFs = np.empty(len(fb_paths))
    ns = np.empty((len(Deltas), len(EFs)))
    Ds = np.empty((len(Deltas), len(EFs)))
    alphas = np.empty((len(Deltas), len(EFs), 2))

    for i, bs_path in enumerate(bs_paths):
        start = bs_path.find('_Delta') + 6  # Start of Delta value
        end = bs_path.find('.h5')  # end of Delta value

        Deltas[i] = float(bs_path[start:end])  # extract value

        fb_paths = glob.glob(bs_path[:-3] + '\FilledBands*.h5')
        fb_paths.sort(key=os.path.getmtime)

        for j, fb_path in enumerate(fb_paths):
            start = fb_path.find('_EF') + 3  # start of EF value
            end = fb_path.find('_T') # end of EF value
            EFs[j] = float(fb_path[start:end])

            if parameters_only:
                continue
            if not parameters_only:
                from ..bands import FilledBands

                fb = FilledBands.load(fb_path)
                alphas[i,j] = fb.alpha
                ns[i,j] = fb.n
                Ds[i,j] = fb.D

    s = Saver()
    s.Deltas = Deltas
    s.EFs = EFs
    s.ns = ns
    s.Ds = Ds
    s.alphas = alphas

    filename = os.path.join(path, 'summary.h5')
    s.save(filename)

    if parameters_only:
        return Deltas, EFs, None, None, None
    return Deltas, EFs, ns, Ds, alphas


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
                        v = v[()] # [()] gets values from Dataset
                        if type(v) is np.bytes_:
                            try:
                                v = pickle.loads(v)  # load pickled Splines
                            except:
                                pass
                        obj.__dict__[k] = v

                return obj
            obj = read_layer(f)

        return obj


    def save(self, filename, **kwargs):
        '''
        Saves data to a file. Use this function to wrap either `save_hdf5` or
        `save_npz`. We currently wrap `save_hdf5`.

        kwargs passed to save_hdf5
        '''
        directory = os.path.dirname(filename)
        try:
            os.makedirs(directory)
        except OSError as e:  # check if directory doesn't exist
            if e.errno != errno.EEXIST:
                raise
            pass

        self.save_hdf5(filename, **kwargs)


    def save_hdf5(self, filename, compression=None, skip=[]):
        '''
        Saves data to a compressed .h5 file

        filename: full path of destination file
        compression: (int) specified as kwarg compression_opts for
            h5py.Group.create_dataset
        skip: list of attributes that will not be saved
        '''
        self.obj_filenames = []

        with h5py.File(filename, 'w') as f:
            # Recursively search dictionary structure of the class
            def write_layer(group, obj):
                classname = '{0}.{1}'.format(obj.__class__.__module__,
                                             obj.__class__.__name__)
                group.attrs['class'] = classname
                for k, v in obj.__dict__.items():
                    # Ignore attributes in `skip`
                    if k in skip:
                        continue

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
                            try:  # try pickling
                                s = pickle.dumps(v)
                                group.create_dataset(k, data=np.array(s))
                            except:
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
