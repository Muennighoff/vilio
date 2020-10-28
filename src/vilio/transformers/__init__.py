# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "3.0.2"


# Removed all imports here to solve error: ValueError: Custom>TFBertMainLayer has already been registered to <class 'transformers.modeling_tf_bert.TFBertMainLayer'>