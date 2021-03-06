# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: LAMARCK_ML/datasets/Dataset.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from LAMARCK_ML.data_util import Attribute_pb2 as LAMARCK__ML_dot_data__util_dot_Attribute__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='LAMARCK_ML/datasets/Dataset.proto',
  package='LAMARCK_ML',
  syntax='proto3',
  serialized_pb=_b('\n!LAMARCK_ML/datasets/Dataset.proto\x12\nLAMARCK_ML\x1a$LAMARCK_ML/data_util/Attribute.proto\"`\n\x0c\x44\x61tasetProto\x12\x10\n\x08name_val\x18\x01 \x01(\t\x12\x10\n\x08\x63ls_name\x18\x02 \x01(\t\x12,\n\x08\x61ttr_val\x18\x0b \x03(\x0b\x32\x1a.LAMARCK_ML.AttributeProtob\x06proto3')
  ,
  dependencies=[LAMARCK__ML_dot_data__util_dot_Attribute__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_DATASETPROTO = _descriptor.Descriptor(
  name='DatasetProto',
  full_name='LAMARCK_ML.DatasetProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name_val', full_name='LAMARCK_ML.DatasetProto.name_val', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cls_name', full_name='LAMARCK_ML.DatasetProto.cls_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='attr_val', full_name='LAMARCK_ML.DatasetProto.attr_val', index=2,
      number=11, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=87,
  serialized_end=183,
)

_DATASETPROTO.fields_by_name['attr_val'].message_type = LAMARCK__ML_dot_data__util_dot_Attribute__pb2._ATTRIBUTEPROTO
DESCRIPTOR.message_types_by_name['DatasetProto'] = _DATASETPROTO

DatasetProto = _reflection.GeneratedProtocolMessageType('DatasetProto', (_message.Message,), dict(
  DESCRIPTOR = _DATASETPROTO,
  __module__ = 'LAMARCK_ML.datasets.Dataset_pb2'
  # @@protoc_insertion_point(class_scope:LAMARCK_ML.DatasetProto)
  ))
_sym_db.RegisterMessage(DatasetProto)


# @@protoc_insertion_point(module_scope)
