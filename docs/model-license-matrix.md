# Matriz preliminar de modelos y licencias

Revisión documental: 13 de agosto de 2026. No se descargaron pesos.

| Candidato | Licencia declarada | Componente crítico | Compuerta | Android directo | Decisión preliminar |
|---|---|---|---|---|---|
| SANA 0.6B 512 | Apache-2.0 | Gemma 2 2B IT | Bloqueada | No | Sólo referencia arquitectónica |
| SANA-Sprint 0.6B | Apache-2.0 | Gemma 2 2B IT | Bloqueada | No | Sólo referencia de destilación |
| Würstchen v2 | MIT | OpenCLIP ViT-bigG/14 | Revisión manual | No | Posible maestro |
| FLUX.1-schnell | Apache-2.0 | Pipeline bundled y acceso gated | Aprobación automática preliminar | No | Posible maestro |
| AuraFlow | Apache-2.0 | Pipeline de 65.8 GiB | Revisión manual | No | Descartado por tamaño |
| Kandinsky 2.2 | Apache-2.0 | Prior, CLIP y decoder | Revisión manual | No | Descartado por complejidad |
| SDXL 1.0 | OpenRAIL++-M | Restricciones de uso | Rechazada | No | Excluido |
| PixArt-α 512 | OpenRAIL++-M | Restricciones de uso | Rechazada | No | Excluido |
| DeepFloyd IF | Licencia IF | Términos y uso condicionado | Rechazada | No | Excluido |

## Evidencia principal

### SANA

La tarjeta de [SANA 0.6B](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers)
declara Apache-2.0 para el modelo, 590 millones de parámetros y Gemma 2 2B IT
como encoder. También remite expresamente a los términos y la política de usos
prohibidos de Gemma. El repositorio de
[Gemma 2 2B IT](https://huggingface.co/google/gemma-2-2b-it) requiere aceptar su
licencia. Por la política estricta de AvatarFace, el pipeline completo queda
bloqueado.

La tarjeta de
[SANA-Sprint](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px)
declara Apache-2.0 y generación de un paso, pero usa el mismo encoder Gemma.

### Würstchen

El [prior](https://huggingface.co/warp-ai/wuerstchen-prior) y el
[decoder](https://huggingface.co/warp-ai/wuerstchen) declaran MIT. La
arquitectura usa compresión espacial 42×, una propiedad interesante para un
estudiante móvil. Sin embargo, el paquete publicado ocupa aproximadamente 11.4
GiB, depende de CLIP ViT-bigG/14 y su propia tarjeta reconoce pérdida de detalle
en rostros. Se requiere fijar y auditar la revisión exacta del encoder antes de
aprobarlo como maestro.

### FLUX.1-schnell

[FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
declara Apache-2.0 y ofrece una revisión identificable. El checkpoint principal
publicado ocupa aproximadamente 23.8 GB, por lo que no es candidato Android.
Puede considerarse como maestro, pero la descarga es gated y exige revisar las
condiciones presentadas a la cuenta antes de aceptarlas.

### AuraFlow y Kandinsky

[AuraFlow](https://huggingface.co/fal/AuraFlow) declara Apache-2.0, pero el
repositorio publicado ronda 65.8 GB. Se descarta del primer benchmark por costo
de transferencia y memoria.

El decoder de [Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)
declara Apache-2.0 y alrededor de mil millones de parámetros, pero necesita
prior, CLIP y decoder. La complejidad y tamaño exceden el objetivo móvil.

### Modelos rechazados

[SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
y [PixArt-α](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512) usan
OpenRAIL++-M, que incluye restricciones de uso y alcanza derivados por
destilación. No cumplen la política del proyecto.

[DeepFloyd IF](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) usa una licencia
propia, requiere aceptación y describe el uso directo como investigación. Se
excluye.

## Resultado automático actual

El manifiesto versionado se evalúa con:

```bash
avatar-face audit-candidates --json
```

Sólo FLUX.1-schnell pasa actualmente las reglas automáticas, pero aún necesita
revisión manual de las condiciones gated. Ningún candidato cabe directamente
en el presupuesto Android.
