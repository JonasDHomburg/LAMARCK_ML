# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: LAMARCK_ML/reproduction/Ancestry.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='LAMARCK_ML/reproduction/Ancestry.proto',
  package='LAMARCK_ML',
  syntax='proto3',
  serialized_pb=_b('\n&LAMARCK_ML/reproduction/Ancestry.proto\x12\nLAMARCK_ML\"F\n\rAncestryProto\x12\x0e\n\x06method\x18\x01 \x01(\t\x12\x12\n\ndescendant\x18\x02 \x01(\t\x12\x11\n\tancestors\x18\x03 \x03(\tb\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_ANCESTRYPROTO = _descriptor.Descriptor(
  name='AncestryProto',
  full_name='LAMARCK_ML.AncestryProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='method', full_name='LAMARCK_ML.AncestryProto.method', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='descendant', full_name='LAMARCK_ML.AncestryProto.descendant', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ancestors', full_name='LAMARCK_ML.AncestryProto.ancestors', index=2,
      number=3, type=9, cpp_type=9, label=3,
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
  serialized_start=54,
  serialized_end=124,
)

DESCRIPTOR.message_types_by_name['AncestryProto'] = _ANCESTRYPROTO

AncestryProto = _reflection.GeneratedProtocolMessageType('AncestryProto', (_message.Message,), dict(
  DESCRIPTOR = _ANCESTRYPROTO,
  __module__ = 'LAMARCK_ML.reproduction.Ancestry_pb2'
  # @@protoc_insertion_point(class_scope:LAMARCK_ML.AncestryProto)
  ))
_sym_db.RegisterMessage(AncestryProto)


# @@protoc_insertion_point(module_scope)