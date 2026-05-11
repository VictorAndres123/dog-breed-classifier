import keras
import keras.layers
import tensorflow as tf

orig_bn = keras.layers.BatchNormalization.__init__
def patched_bn(self, **kwargs):
    kwargs.pop('renorm', None)
    kwargs.pop('renorm_clipping', None)
    kwargs.pop('renorm_momentum', None)
    orig_bn(self, **kwargs)
keras.layers.BatchNormalization.__init__ = patched_bn

orig_rf = keras.layers.RandomFlip.__init__
def patched_rf(self, **kwargs):
    kwargs.pop('data_format', None)
    orig_rf(self, **kwargs)
keras.layers.RandomFlip.__init__ = patched_rf

orig_rr = keras.layers.RandomRotation.__init__
def patched_rr(self, **kwargs):
    kwargs.pop('data_format', None)
    orig_rr(self, **kwargs)
keras.layers.RandomRotation.__init__ = patched_rr

orig_rz = keras.layers.RandomZoom.__init__
def patched_rz(self, **kwargs):
    kwargs.pop('data_format', None)
    orig_rz(self, **kwargs)
keras.layers.RandomZoom.__init__ = patched_rz

orig_dense = keras.layers.Dense.__init__
def patched_dense(self, **kwargs):
    kwargs.pop('quantization_config', None)
    orig_dense(self, **kwargs)
keras.layers.Dense.__init__ = patched_dense

orig_input = keras.layers.InputLayer.__init__
def patched_input(self, **kwargs):
    kwargs.pop('optional', None)
    batch_shape = kwargs.pop('batch_shape', None)
    if batch_shape is not None and 'shape' not in kwargs:
        kwargs['shape'] = batch_shape[1:]
    orig_input(self, **kwargs)
keras.layers.InputLayer.__init__ = patched_input

print('TF:', tf.__version__)
print('Keras:', keras.__version__)

model = keras.models.load_model(
    'models/dog_classifier.keras',
    compile=False
)

model.save('models/dog_classifier_fixed.keras')
print('Modelo guardado correctamente')