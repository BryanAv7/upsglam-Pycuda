# app.py
from flask import Flask, request, jsonify
import tempfile
import os
import pycuda.autoinit

from gpu_filters_rgb import run_gpu_filter_rgb
from new_filter import NewFilter

app = Flask(__name__)

# ==========================================================
# MAPEO DE PARÁMETROS POR FILTRO
# ==========================================================
GPU_FILTERS = {
    "emboss": ["offset", "factor"],
    "sobel": ["factor"],
    "gaussiano": ["sigma"],
    "sharpen": ["sharp_factor"]
}

PY_FILTERS = {
    "sombras_epico": ["highlight_boost", "vignette_strength"],
    "resaltado_frio": ["blue_boost", "contrast"]
}

GPU_DEFAULTS = {
    "factor": 2.0,
    "offset": 128.0,
    "sigma": 90.0,
    "sharp_factor": 20.0
}

PY_DEFAULTS = {
    "highlight_boost": 1.1,
    "vignette_strength": 0.5,
    "blue_boost": 1.2,
    "contrast": 1.3
}

# ==========================================================
#           ENDPOINT PRINCIPAL
# ==========================================================
@app.route("/procesar", methods=["POST"])
def process_image():
    temp_input = None
    temp_output = None

    try:
        if "imagen" not in request.files:
            return jsonify({"error": "Debe enviar una imagen"}), 400

        image = request.files["imagen"]
        filter_name = request.form.get("filtro", "").lower()

        # Validación de extensión
        ext = os.path.splitext(image.filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg"]:
            return jsonify({"error": "Formato de imagen no soportado"}), 400

        # Guardar archivo temporal (MISMO FORMATO)
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_input.close()
        image.save(temp_input.name)

        # ==========================================================
        # 1️⃣ FILTROS GPU
        # ==========================================================
        if filter_name in GPU_FILTERS:

            params = {
                p: float(request.form.get(p, GPU_DEFAULTS[p]))
                for p in GPU_FILTERS[filter_name]
            }

            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            temp_output.close()

            elapsed, (W, H) = run_gpu_filter_rgb(
                input_path=temp_input.name,
                output_path=temp_output.name,
                filter_name=filter_name,
                ksize=3,
                factor=params.get("factor", 2.0),
                offset=params.get("offset", 128.0),
                sigma=params.get("sigma", 90.0),
                sharp_factor=params.get("sharp_factor", 20.0),
                block=(16, 16, 1)
            )

            # Devolver imagen en BYTES
            with open(temp_output.name, "rb") as f:
                processed_bytes = f.read()

            return processed_bytes, 200, {
                "Content-Type": f"image/{ext.replace('.', '')}",
                "Content-Disposition": f"inline; filename=procesada{ext}"
            }

        # ==========================================================
        # 2️⃣ FILTROS PYTHON — BASADOS EN ARCHIVOS
        # ==========================================================
        elif filter_name in PY_FILTERS:

            params = {
                p: float(request.form.get(p, PY_DEFAULTS[p]))
                for p in PY_FILTERS[filter_name]
            }

            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            temp_output.close()

            # Selección del filtro Python
            if filter_name == "sombras_epico":
                NewFilter.filtro_sombras_epico_path(
                    temp_input.name,
                    temp_output.name,
                    highlight_boost=params["highlight_boost"],
                    vignette_strength=params["vignette_strength"]
                )

            elif filter_name == "resaltado_frio":
                NewFilter.filtro_resaltado_frio_path(
                    temp_input.name,
                    temp_output.name,
                    blue_boost=params["blue_boost"],
                    contrast=params["contrast"]
                )

            # Respuesta final en BYTES
            with open(temp_output.name, "rb") as f:
                processed_bytes = f.read()

            return processed_bytes, 200, {
                "Content-Type": f"image/{ext.replace('.', '')}",
                "Content-Disposition": f"inline; filename=procesada{ext}"
            }

        else:
            return jsonify({"error": f"Filtro '{filter_name}' no existe"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_input and os.path.exists(temp_input.name):
            os.unlink(temp_input.name)
        if temp_output and os.path.exists(temp_output.name):
            os.unlink(temp_output.name)


# ==========================================================
# INICIO DEL SERVIDOR
# ==========================================================
if __name__ == "__main__":
    print("Iniciando servidor Flask ")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
