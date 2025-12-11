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
