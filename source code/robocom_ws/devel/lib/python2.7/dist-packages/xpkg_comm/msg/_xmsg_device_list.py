# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from xpkg_comm/xmsg_device_list.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import xpkg_comm.msg

class xmsg_device_list(genpy.Message):
  _md5sum = "2ceb7f0b4db86b40356e12aab2f69511"
  _type = "xpkg_comm/xmsg_device_list"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """uint8 dev_count
xmsg_device[] dev_list

================================================================================
MSG: xpkg_comm/xmsg_device
uint8 dev_class
uint8 dev_type
uint8 dev_number
uint8 dev_enable
"""
  __slots__ = ['dev_count','dev_list']
  _slot_types = ['uint8','xpkg_comm/xmsg_device[]']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       dev_count,dev_list

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(xmsg_device_list, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.dev_count is None:
        self.dev_count = 0
      if self.dev_list is None:
        self.dev_list = []
    else:
      self.dev_count = 0
      self.dev_list = []

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      buff.write(_get_struct_B().pack(self.dev_count))
      length = len(self.dev_list)
      buff.write(_struct_I.pack(length))
      for val1 in self.dev_list:
        _x = val1
        buff.write(_get_struct_4B().pack(_x.dev_class, _x.dev_type, _x.dev_number, _x.dev_enable))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.dev_list is None:
        self.dev_list = None
      end = 0
      start = end
      end += 1
      (self.dev_count,) = _get_struct_B().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.dev_list = []
      for i in range(0, length):
        val1 = xpkg_comm.msg.xmsg_device()
        _x = val1
        start = end
        end += 4
        (_x.dev_class, _x.dev_type, _x.dev_number, _x.dev_enable,) = _get_struct_4B().unpack(str[start:end])
        self.dev_list.append(val1)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      buff.write(_get_struct_B().pack(self.dev_count))
      length = len(self.dev_list)
      buff.write(_struct_I.pack(length))
      for val1 in self.dev_list:
        _x = val1
        buff.write(_get_struct_4B().pack(_x.dev_class, _x.dev_type, _x.dev_number, _x.dev_enable))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.dev_list is None:
        self.dev_list = None
      end = 0
      start = end
      end += 1
      (self.dev_count,) = _get_struct_B().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.dev_list = []
      for i in range(0, length):
        val1 = xpkg_comm.msg.xmsg_device()
        _x = val1
        start = end
        end += 4
        (_x.dev_class, _x.dev_type, _x.dev_number, _x.dev_enable,) = _get_struct_4B().unpack(str[start:end])
        self.dev_list.append(val1)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_B = None
def _get_struct_B():
    global _struct_B
    if _struct_B is None:
        _struct_B = struct.Struct("<B")
    return _struct_B
_struct_4B = None
def _get_struct_4B():
    global _struct_4B
    if _struct_4B is None:
        _struct_4B = struct.Struct("<4B")
    return _struct_4B
