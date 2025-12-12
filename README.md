# Servidor de PyCUDA

Esta es una API RESTful para procesamiento de imágenes con aceleración por GPU mediante **PyCUDA**, combinada con filtros clásicos implementados en NumPy/PIL que sera utilizado en nuestra aplicación UPSGLAM 2.0.

> Soporta filtros en **GPU** (Sobel, Emboss, Gauss, Sharpen) y en **CPU** (Sombras Épico, Resaltado Frío, Marco).

---

##  Arquitectura General

```text
app.py
├─  Flask API: `/procesar` (POST)
│  ├─ Recibe imagen (PNG/JPG)
│  ├─ Aplica filtro según parámetros
│  └─ Devuelve imagen procesada (bytes)
│
├─  Filtros GPU (pycuda)
│  └─ `gpu_filters_rgb.py`: kernels CUDA + wrappers Python
│
├─  Filtros CPU (NumPy/PIL)
│  └─ `new_filter.py`: clase `NewFilter` con métodos vectorizados y basados en archivo
│
└─ Recursos externos
   └─ `marcos/vertical.png`, `marcos/horizontal.png` (requeridos para `marco`)
```

---

## Mapeo de los Filtros

```

| Filtro             | Tipo | Parámetros (con valores por defecto)                         | Descripción |
|--------------------|------|--------------------------------------------------------------|-------------|
| `emboss`           | GPU  | `offset=128.0`, `factor=2.0`                                 | Efecto 3D de relieve |
| `sobel`            | GPU  | `factor=2.0`                                                 | Detección de bordes |
| `gauss`            | GPU  | `sigma=90.0`                                                 | Desenfoque suave |
| `sharpen`          | GPU  | `sharp_factor=20.0`                                          | Realce de bordes |
| `sombras_epico`    | CPU  | `highlight_boost=1.1`, `vignette_strength=0.5`              | Estilo cinematográfico: sombras cálidas, luces frías + viñeteo |
| `resaltado_frio`   | CPU  | `blue_boost=1.2`, `contrast=1.3`                            | Tonos fríos realzados (azul/teal) |
| `marco`            | CPU  | — (sin parámetros)                                           | Superpone marco PNG adaptativo (vertical/horizontal) |

```

---

## Requisitos

1. Entorno
- Windows (recomendado) con controladores NVIDIA y CUDA Toolkit instalados
- Python ≥ 3.9
- NVIDIA GPU con soporte CUDA (Compute Capability ≥ 3.5)

2. Dependencias (requirements.txt)
   
```
flask==3.0.*
pycuda==2024.1
numpy==1.26.*
pillow==10.*
```

- Instalación:

```
pip install -r requirements.txt
```

---

## Despliegue

- Local
  
```
python app.py
# Servidor en http://localhost:5000
```

- Producción

```
  gunicorn --workers=1 --threads=1 --bind 0.0.0.0:5000 app:app
```

---

## Uso de la API

- Endpoint
Es el encargado de procesar la imagen que el usuario haya cargado.

```
POST /procesar
```

Parámetros: 
- imagen (archivo): imagen en PNG, JPG o JPEG (obligatorio)
- filtro (string): nombre del filtro (obligatorio)
- Parámetros específicos del filtro (opcionales):
factor, offset, sigma, sharp_factor, highlight_boost, vignette_strength, blue_boost, contrast



