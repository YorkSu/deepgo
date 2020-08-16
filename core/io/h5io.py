# -*- coding: utf-8 -*-
"""HDF5 IO
  ======

  Deep Go HDF5 IO Class
"""


import os
import h5py

from deepgo.core import exception
from deepgo.core.abcs import IO
from deepgo.core.api import np


class H5pyIO(IO):
  """HDF5 IO
    ======
  
    Deep Go HDF5 IO Class
  """
  @staticmethod
  def assert_len(obj, index):
    """Assert Length

      Ensure index is accessible within Object

      Args:
        obj: h5py.Dataset
        index: Integer.

      Raises:
        DeepGo.TypeError
    """
    if index < -len(obj) or index >= len(obj):
      raise exception.TypeError(", ".join([
          f"Index out of range",
          f"excepted -{len(obj)} <= index < {len(obj)}",
          f"got {index}"]))

  @staticmethod
  def close(f):
    """Close the Object if h5py.File
    
      Args:
        f: h5py.File
    """
    if isinstance(f, h5py.File):
      f.close()

  @staticmethod
  def create_dataset(
      group,
      name,
      shape=(),
      dtype=np.uint8,
      use_compression=True,
      use_chunk=True,
      maxlen=None,
      compression='gzip',
      compression_opts=4):
    """Create a Dataset object within Group/File

      Args:
        group: h5py.Group or h5py.File
        name: String, name of the dataset
        shape: Tuple, shape of the dataset without the first dimension
        dtype: Dtype, dtype of the dataset
        use_compression: Boolean, whether to compress the data
        use_chunk: Boolean, whether to use chunks
        maxlen: Integer, maximum length of the dataset
        compression: String, compression method
        compression_opts: Integer, compression parameters
    
      Returns:
        h5py.Dataset
    """
    exception.assert_h5_group(group)
    kwargs = {}
    if use_compression:
      kwargs['compression'] = compression
      kwargs['compression_opts'] = compression_opts
    if use_chunk:
      kwargs['chunks'] = (1, *shape)
    if maxlen is not None:
      kwargs['shape'] = (maxlen, *shape)
    else:
      kwargs['shape'] = (1, *shape)
      kwargs['maxshape'] = (None, *shape)
    return group.create_dataset(
        name,
        dtype=dtype,
        **kwargs)

  @staticmethod
  def create_group(group, name):
    """Create a Group object within Group/File

      Args:
        group: h5py.Group or h5py.File
        name: String, name of the Group
    """
    exception.assert_h5_group(group)
    return group.create_group(name)

  @staticmethod
  def filename(name='', path=None, suffix='.h5'):
    """Filename Process

      Description:
        If the path does not exist, it is created automatically.
        If the file name does not have a suffix, it is automatically replenished.

      Args:
        name: String, name of the file
        path: String or None, path to the file
        suffix: String, the extension name of the file
    
      Returns:
        String, the full name of the file
    """
    if path is None:
      _filename = name
    else:
      _filename = os.path.join(path, name)
    _path, _file = os.path.split(_filename)
    _name, ext = os.path.splitext(_file)
    if not os.path.exists(_path):
      os.makedirs(_path)
    if ext != suffix:
      _filename = os.path.join(_path, _name) + suffix
    return _filename

  @staticmethod
  def get_attrs(group):
    """Get Attributes from Group/File

      Args:
        group: h5py.Group or h5py.File

      Returns:
        Dict
    """
    exception.assert_h5_group(group)
    return {key: value for key, value in group.attrs.items()}

  @staticmethod
  def keys(group):
    """Get keys from Group/File

      The same as `group.keys()`

      Args:
        group: h5py.Group or h5py.File

      Returns:
        List, keys of the group
    """
    exception.assert_h5_group(group)
    return group.keys()

  @staticmethod
  def open(filename, mode='a'):
    """Open a h5py.File with filename

      Args:
        filename: String, the filename of the HDF5 file
        mode: String, file open mode

      Returns:
        h5py.File
    """
    return h5py.File(filename, mode=mode)

  @staticmethod
  def set_attrs(group, dictionary):
    """Set key-value pairs within Group from a dictionary

      Args:
        group: h5py.Group or h5py.File
        dictionary: Dict
    """
    exception.assert_h5_group(group)
    for key, value in dictionary.items():
      group.attrs[key] = value


assert_len = H5pyIO.assert_len
close = H5pyIO.close
create_dataset = H5pyIO.create_dataset
create_group = H5pyIO.create_group
filename = H5pyIO.filename
get_attrs = H5pyIO.get_attrs
keys = H5pyIO.keys
open = H5pyIO.open
set_attrs = H5pyIO.set_attrs


class H5pyDatasetIO(IO):
  """HDF5 Dataset IO
    ======
  
    Deep Go HDF5 Dataset IO Class
  """
  @staticmethod
  def append(dset, data):
    """Add Data at the end of the Dataset

      Args:
        dset: h5py.Dataset
        data: np.ndarray, shape=dset.shape[1:]
    """
    exception.assert_h5_dataset(dset)
    H5pyDatasetIO.relen(dset, len(dset) + 1)
    dset[-1] = data

  @staticmethod
  def batch_fill(dset, datas, offset):
    """Fill Datas from offset of the Dataset

      Args:
        dset: h5py.Dataset
        datas: np.ndarray, shape=(n, *dset.shape[1:])
        offset: Integer, the location where the fill begins
    """
    exception.assert_h5_dataset(dset)
    length = len(datas)
    if offset + length > len(dset):
      H5pyDatasetIO.relen(dset, offset + length)
    dset[offset:offset + length] = datas

  @staticmethod
  def empty(dset, length, dtype=None):
    """Create a empty np.ndarray like Dataset

      Args:
        dset: h5py.Dataset
        length: Integer, the length of the np.ndarray
        dtype: Dtype, dtype of the np.ndarray

      Returns:
        np.ndarray, shape(length, *dset.shape[1:])
    """
    exception.assert_h5_dataset(dset)
    if dtype is None:
      dtype = dset.dtype
    return np.empty((length, *H5pyDatasetIO.shape(dset)), dtype=dtype)

  @staticmethod
  def extend(dset, datas):
    """Extend Datas at the end of the Dataset

      Args:
        dset: h5py.Dataset
        datas: np.ndarray, shape=(n, *dset.shape[1:])
    """
    exception.assert_h5_dataset(dset)
    offset = len(dset)
    length = len(datas)
    H5pyDatasetIO.relen(dset, offset + length)
    dset[offset:offset + length] = datas

  @staticmethod
  def get(dset, index):
    """Get a Data from Dataset from the given index

      Args:
        dset: h5py.Dataset
        index: Integer. Index of the Data

      Returns:
        np.ndarray, shape(*dset.shape[1:])
    """
    exception.assert_h5_dataset(dset)
    assert_len(dset, index)
    return dset[index]

  @staticmethod
  def len(dset):
    """Get the first dimension shape of the Dataset

      Args:
        dset: h5py.Dataset
      
      Return:
        Integer, dset.shape[0]
    """
    exception.assert_h5_dataset(dset)
    return len(dset)

  @staticmethod
  def read_direct(dset, receiver, start=None, stop=None):
    """Get Datas from the HDF5 file directly

      Args:
        dset: h5py.Dataset
        receiver: np.ndarray. Receive the datas.
        start: Integer, Optional.
        stop: Integer, Optional.

      Changes:
        1. If start and stop
          Dataset[start:stop] to receiver
        2. If start
          Dataset[start:] to receiver
        3. If stop
          Dataset[:stop] to receiver
        4. Else
          Dataset to receiver
    """
    exception.assert_h5_dataset(dset)
    if start is not None and stop is not None:
      dset.read_direct(receiver, np.s_[start:stop])
    elif start is not None:
      dset.read_direct(receiver, np.s_[start:])
    elif stop is not None:
      dset.read_direct(receiver, np.s_[:stop])
    else:
      dset.read_direct(receiver)

  @staticmethod
  def relen(dset, length):
    """Reset the Dataset length

      Args:
        dset: h5py.Dataset
        length: Integer. The new length of the Dataset
    """
    exception.assert_h5_dataset(dset)
    H5pyDatasetIO.reshape(dset, (length, *dset.shape[1:]))

  @staticmethod
  def reshape(dset, shape):
    """Reset the Dataset shape

      Args:
        dset: h5py.Dataset
        shape: Tuple of Integer. The new shape of the Dataset

      Alias:
        resize
    """
    exception.assert_h5_dataset(dset)
    dset.resize(shape)

  resize = reshape

  @staticmethod
  def set(dset, index, data):
    """Set a Data from Dataset from the given index

      Args:
        dset: h5py.Dataset
        index: Integer. Index of the Data
        data: np.ndarray, shape=dset.shape[1:]
      
      Alias:
        replace
    """
    exception.assert_h5_dataset(dset)
    assert_len(dset, index)
    dset[index] = data

  replace = set

  @staticmethod
  def shape(dset):
    """Get the shape of the Dataset

      Args:
        dset: h5py.Dataset

      Returns:
        Tuple, dset.shape[1:]
    """
    exception.assert_h5_dataset(dset)
    return dset.shape[1:]

  @staticmethod
  def slice(dset, start, stop, dtype=None):
    """Get a np.ndarray from a slice of Dataset

      Args:
        dset: h5py.Dataset
        start: Integer, the start of the slice
        stop: Integer, the stop of the slice
        dtype: Dtype, dtype of the np.ndarray

      Returns:
        np.ndarray, shape(stop - start, *dset.shape[1:])
    """
    exception.assert_h5_dataset(dset)
    receiver = H5pyDatasetIO.zeros(dset, stop - start, dtype)
    H5pyDatasetIO.read_direct(dset, receiver, start, stop)
    return receiver

  @staticmethod
  def zeros(dset, length, dtype=None):
    """Create a zero np.ndarray like Dataset

      Args:
        dset: h5py.Dataset
        length: Integer, the length of the np.ndarray
        dtype: Dtype, dtype of the np.ndarray

      Returns:
        np.ndarray, shape(length, *dset.shape[1:])
    """
    exception.assert_h5_dataset(dset)
    if dtype is None:
      dtype = dset.dtype
    return np.zeros((length, *H5pyDatasetIO.shape(dset)), dtype=dtype)


# Aliases
ds = H5pyDatasetIO
dataset = H5pyDatasetIO


append = H5pyDatasetIO.append
batch_fill = H5pyDatasetIO.batch_fill
empty = H5pyDatasetIO.empty
extend = H5pyDatasetIO.extend
get = H5pyDatasetIO.get
len = H5pyDatasetIO.len
read_direct = H5pyDatasetIO.read_direct
relen = H5pyDatasetIO.relen
replace = H5pyDatasetIO.replace
reshape = H5pyDatasetIO.reshape
resize = H5pyDatasetIO.resize
set = H5pyDatasetIO.set
shape = H5pyDatasetIO.shape
slice = H5pyDatasetIO.slice
zeros = H5pyDatasetIO.zeros

