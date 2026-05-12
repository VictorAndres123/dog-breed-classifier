"""
fix_model.py — Parche completo para modelos guardados con Keras 3.x
cargados en Keras 2.14
"""

import os, sys, zipfile, json, tempfile, shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import keras

print(f"TensorFlow : {tf.__version__}")
print(f"Keras      : {keras.__version__}")
print()

MODEL_KERAS = "models/dog_classifier_fixed.keras"
MODEL_H5    = "models/dog_classifier_fixed.h5"

# ── Parche del config.json dentro del .keras (ZIP) ───────────────────────────

def patch_config(cfg_str: str) -> str:
    """Aplica todos los parches necesarios para Keras 3→2 al JSON de config."""

    # 1. batch_shape → batch_input_shape en InputLayer
    cfg_str = cfg_str.replace('"batch_shape"', '"batch_input_shape"')

    # 2. Quitar 'data_format' de RandomFlip (no existe en Keras 2)
    try:
        cfg = json.loads(cfg_str)

        def fix_layer(layer_cfg):
            """Recorre recursivamente y parchea cada capa."""
            if not isinstance(layer_cfg, dict):
                return layer_cfg

            class_name = layer_cfg.get("class_name", "")
            config = layer_cfg.get("config", {})

            # RandomFlip: quitar data_format
            if class_name == "RandomFlip" and "data_format" in config:
                del config["data_format"]

            # RandomRotation: quitar data_format
            if class_name == "RandomRotation" and "data_format" in config:
                del config["data_format"]

            # RandomZoom: quitar data_format
            if class_name == "RandomZoom" and "data_format" in config:
                del config["data_format"]

            # RandomTranslation: quitar data_format
            if class_name == "RandomTranslation" and "data_format" in config:
                del config["data_format"]

            # DTypePolicy → string simple
            if isinstance(config.get("dtype"), dict):
                dtype_val = config["dtype"].get("config", {}).get("name", "float32")
                config["dtype"] = dtype_val

            # Recursión en sub-listas (layers anidadas)
            for key, val in config.items():
                if isinstance(val, list):
                    config[key] = [fix_layer(v) for v in val]
                elif isinstance(val, dict):
                    config[key] = fix_layer(val)

            # Recursión en layers del modelo
            if "layers" in layer_cfg:
                layer_cfg["layers"] = [fix_layer(l) for l in layer_cfg["layers"]]

            return layer_cfg

        cfg = fix_layer(cfg)
        return json.dumps(cfg)

    except Exception as e:
        print(f"  ⚠️  Error parcheando JSON: {e}")
        return cfg_str


print("⏳ Abriendo modelo y aplicando parches al config.json...")

tmp_dir = tempfile.mkdtemp()
patched = os.path.join(tmp_dir, "model_patched.keras")

try:
    with zipfile.ZipFile(MODEL_KERAS, "r") as zin:
        names = zin.namelist()
        print(f"   Archivos dentro del .keras: {names}")

        with zipfile.ZipFile(patched, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)

                if item.filename.endswith(".json"):
                    cfg_str = data.decode("utf-8")
                    cfg_str = patch_config(cfg_str)
                    data = cfg_str.encode("utf-8")
                    print(f"   ✅ Parcheado: {item.filename}")

                zout.writestr(item, data)

    print("✅ Parche aplicado\n")

except Exception as e:
    print(f"❌ Error al parchear el ZIP: {e}")
    shutil.rmtree(tmp_dir)
    sys.exit(1)

# ── Intentar cargar el modelo parcheado ──────────────────────────────────────

model = None

for label, kwargs in [
    ("normal",          {}),
    ("custom_objects",  {"custom_objects": {"InputLayer": tf.keras.layers.InputLayer}}),
]:
    if model is not None:
        break
    try:
        print(f"⏳ Cargando modelo parcheado ({label})...")
        model = tf.keras.models.load_model(patched, compile=False, **kwargs)
        print(f"✅ Carga exitosa ({label})")
    except Exception as e:
        print(f"❌ Falló ({label}): {e}\n")

shutil.rmtree(tmp_dir)

if model is None:
    print()
    print("━" * 60)
    print("🚨 El parche no fue suficiente.")
    print()
    print("La solución definitiva es re-guardar el modelo en el")
    print("entorno donde fue entrenado (Keras 3.x) con:")
    print()
    print("   import tensorflow as tf")
    print('   model = tf.keras.models.load_model("models/dog_classifier_fixed.keras")')
    print('   model.save("models/dog_classifier_fixed.h5")')
    print()
    print("O instala Keras 3 en este entorno:")
    print("   pip install keras==3.3.3 tensorflow==2.16.1")
    print("━" * 60)
    sys.exit(1)

# ── Verificar y guardar ───────────────────────────────────────────────────────

print()
print(f"📐 Input shape  : {model.input_shape}")
print(f"📐 Output shape : {model.output_shape}")
print(f"🐶 Clases       : {model.output_shape[-1]}")

dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
pred  = model.predict(dummy, verbose=0)
print(f"✅ Test predicción OK — shape: {pred.shape}")

print()
print(f"💾 Guardando como '{MODEL_H5}'...")
model.save(MODEL_H5)
print(f"✅ Guardado exitoso")
print()
print("━" * 60)
print("Actualiza app.py, cambia la ruta del modelo a:")
print('   "models/dog_classifier_fixed.h5"')
print("━" * 60)