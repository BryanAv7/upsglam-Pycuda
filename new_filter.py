# new_filter.py
import numpy as np
from PIL import Image

class NewFilter:

    # ============================================================
    #                MÉTODOS ORIGINALES (NO TOCADOS)
    # ============================================================

    @staticmethod
    def filtro_sombras_epico(
        img_np,
        highlight_boost=1.1,
        vignette_strength=0.5
    ):
        img = img_np.astype(np.float32) / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # 1. S-curve contrast
        lumin = 0.299 * r + 0.587 * g + 0.114 * b
        lumin = np.clip((lumin - 0.5) * 1.5 + 0.5, 0, 1)

        avg = (r + g + b) / 3 + 1e-6
        r = np.clip(r * lumin / avg, 0, 1)
        g = np.clip(g * lumin / avg, 0, 1)
        b = np.clip(b * lumin / avg, 0, 1)

        # 2. Shadows & lights
        shadow_mask = lumin < 0.5
        light_mask = lumin >= 0.5

        r[shadow_mask] *= 0.7
        g[shadow_mask] *= 0.7
        b[shadow_mask] *= 0.85

        r[light_mask] = np.clip(r[light_mask] * highlight_boost + 0.05, 0, 1)
        g[light_mask] = np.clip(g[light_mask] * (highlight_boost - 0.05) + 0.05, 0, 1)
        b[light_mask] = np.clip(b[light_mask] * (highlight_boost - 0.15), 0, 1)

        # 3. Teal & Orange
        b = np.clip(b + (0.25 - lumin) * 0.3, 0, 1)
        r = np.clip(r + lumin * 0.1, 0, 1)

        # 4. Vignette
        h, w = r.shape
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        vignette = 1 - vignette_strength * (dist / max_dist)**2

        r *= vignette
        g *= vignette
        b *= vignette

        # 5. Gamma
        gamma = 1 / 1.2
        r = r ** gamma
        g = g ** gamma
        b = b ** gamma

        out = np.stack([r, g, b], axis=-1) * 255
        return np.clip(out, 0, 255).astype(np.uint8)


    @staticmethod
    def filtro_resaltado_frio(
        img_np,
        blue_boost=1.2,
        contrast=1.3
    ):
        img = img_np.astype(np.float32) / 255.0
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

        # 1. Blue tone
        b = np.clip(b * blue_boost, 0, 1)
        g = np.clip(g * (1 + (blue_boost - 1) * 0.4), 0, 1)

        img = np.stack([r, g, b], axis=-1)
        img = np.clip((img - 0.5) * contrast + 0.5, 0, 1)

        return (img * 255).astype(np.uint8)


    # ============================================================
    #            MÉTODOS NUEVOS BASADOS EN ARCHIVOS
    # ============================================================

    @staticmethod
    def filtro_sombras_epico_path(input_path, output_path,
                                  highlight_boost=1.1,
                                  vignette_strength=0.5):
        """
        Carga una imagen desde archivo, aplica el filtro y guarda el resultado.
        """
        pil_img = Image.open(input_path).convert("RGB")
        img_np = np.array(pil_img)

        result_np = NewFilter.filtro_sombras_epico(
            img_np,
            highlight_boost=highlight_boost,
            vignette_strength=vignette_strength
        )

        result_pil = Image.fromarray(result_np)
        result_pil.save(output_path)

        return output_path


    @staticmethod
    def filtro_resaltado_frio_path(input_path, output_path,
                                   blue_boost=1.2,
                                   contrast=1.3):
        """
        Carga una imagen desde archivo, aplica el filtro y guarda el resultado.
        """
        pil_img = Image.open(input_path).convert("RGB")
        img_np = np.array(pil_img)

        result_np = NewFilter.filtro_resaltado_frio(
            img_np,
            blue_boost=blue_boost,
            contrast=contrast
        )

        result_pil = Image.fromarray(result_np)
        result_pil.save(output_path)

        return output_path


    # ============================================================
    #                  FILTRO DE MARCO (NUEVO)
    # ============================================================

    @staticmethod
    def filtro_marco(
        img_np,
        marco_vertical_path,
        marco_horizontal_path
    ):
        """
        Superpone un marco PNG transparente sobre la imagen,
        eligiendo automáticamente el marco según orientación:
        - Vertical  (alto >= ancho)  -> marco_vertical_path
        - Horizontal (ancho > alto)  -> marco_horizontal_path

        El marco se redimensiona al tamaño exacto de la imagen.
        """
        from PIL import Image
        import numpy as np

        # Tamaño de la imagen original
        h, w = img_np.shape[:2]

        # Seleccionar marco según orientación
        marco_path = (
            marco_vertical_path
            if h >= w
            else marco_horizontal_path
        )

        # Cargar marco con canal alpha
        marco = Image.open(marco_path).convert("RGBA")

        # Redimensionar marco al tamaño exacto de la imagen
        marco = marco.resize((w, h), Image.LANCZOS)

        marco_np = np.array(marco)              # (h, w, 4)
        marco_rgb = marco_np[:, :, :3]          # RGB del marco
        marco_alpha = marco_np[:, :, 3:] / 255.0  # Alpha normalizado

        # Convertir imagen base a float32
        base = img_np.astype(np.float32)

        # Superposición
        result = base * (1 - marco_alpha) + marco_rgb * marco_alpha

        return np.clip(result, 0, 255).astype(np.uint8)


    @staticmethod
    def filtro_marco_path(
        input_path,
        output_path,
        marco_vertical_path,
        marco_horizontal_path
    ):
        """
        Carga imagen, aplica el filtro de marco y guarda el resultado.
        Funciona igual que los demás filtros _path.
        """
        from PIL import Image
        import numpy as np

        # Cargar la imagen base en RGB
        pil_img = Image.open(input_path).convert("RGB")
        img_np = np.array(pil_img)

        # Aplicar filtro
        result_np = NewFilter.filtro_marco(
            img_np,
            marco_vertical_path,
            marco_horizontal_path
        )

        # Guardar
        result_pil = Image.fromarray(result_np)
        result_pil.save(output_path)

        return output_path
