# Plan de LoRA en GPU — release v2

## Base seleccionada

Se selecciona `warp-ai/wuerstchen-prior-model-base` (Stage C, v2-base), revisión
`3f9205c8c2e7cf103192954fe6f096e66f9d4efc`, bajo MIT. El decoder Würstchen y
el encoder OpenCLIP quedan congelados y se registran con sus commits en
`configs/model-candidates.json`. Esta separación permite adaptar sólo el prior
de aproximadamente 1B de parámetros, manteniendo el alcance dentro de una RTX
4090 de 48 GiB.

La descarga local, hashes y paquete de transferencia están registrados en
`docs/model-download-wuerstchen-v2.md`. El paquete es mayor a 100 MB, por lo que
se transfiere por Drive; no se descargan pesos directamente en Vast.

## Alcance de la primera corrida

- Datos: release congelada `v2.0.0`, manifiesto SHA-256
  `79ecdd3f36301c4462372be35e93f66cee3e52f51d6992050728da8dc84334a2`.
- Resolución: 256 × 256; sin augmentations en la primera corrida.
- Adaptación: LoRA sólo sobre las capas de atención del transformer; base
  congelada.
- Parámetros iniciales: rank 16, alpha 16, dropout 0.05, batch efectivo 4,
  precisión bf16, gradient checkpointing y acumulación de gradiente si hiciera
  falta.
- Validación: conservar `validation` y `test` declarados; nunca mezclarlos con
  train ni recalcular splits.

## Compuertas antes de pagar entrenamiento

1. Confirmar la licencia/NOTICE del commit exacto y conservarlos fuera de Git
   junto al manifiesto de descarga.
2. Descargar sólo en local, fijar hashes de todos los pesos y empaquetar según
   la regla de 100 MB.
3. En Vast, verificar release v2, CUDA, hashes de pesos y espacio libre antes
   de iniciar la corrida.
4. Ejecutar un piloto de 20 pasos con límite explícito de coste/tiempo y guardar
   checkpoint, config, seed, pérdidas y muestras de validation.
5. Revisar visualmente las muestras y evaluar fuga de duplicados antes de
   escalar. El LoRA resultante pasa de nuevo por la compuerta de licencia.

El objetivo de esta fase es validar adaptación y trazabilidad, no lanzar un
entrenamiento de producción ni convertir FLUX en el runtime Android final.

## Preflight de implementación en Vast (2026-08-14)

El checkout clonado quedó en `063ff62`, con el dataset `v2.0.0` aprobado y los
pesos enlazados desde la copia previamente auditada. La instancia usa PyTorch
`2.12.0+cu130`; se instalaron para el piloto `diffusers==0.39.0`,
`transformers==5.15.0`, `peft==0.20.0`, `accelerate==1.14.0` y
`safetensors==0.8.0`.

En esta versión la clase Würstchen está bajo el módulo deprecado de Diffusers,
por lo que el piloto debe importar explícitamente
`diffusers.pipelines.deprecated.wuerstchen.modeling_wuerstchen_prior.WuerstchenPrior`
y registrar esa dependencia como parte de su configuración reproducible. El
prior local cargó correctamente con 993,636,896 parámetros. La inyección LoRA
con rango/alpha 16 y dropout 0.05 sobre `to_q`, `to_k`, `to_v` y `to_out.0`
dejó 6,291,456 parámetros entrenables (0.6292 %); no se ejecutó optimización.

Durante esta comprobación apareció una incompatibilidad del loader genérico de
Transformers: el índice histórico `pytorch_model.bin.index.json` del encoder
OpenCLIP apunta a `.bin` que no forman parte de la release (se retuvieron sus
shards equivalentes `pytorch_model-*.safetensors`). Ambos shards safetensors
declaran las mismas 1,296 claves del índice. El piloto debe cargarlos mediante
un loader explícito de safetensors de sólo lectura; no se debe editar el índice
ni los pesos congelados para ocultar esta discrepancia.

## Resultado del piloto (2026-08-15)

El piloto de 20 pasos completó forward/backward e inferencia con `lr=1e-4`,
pero la muestra quedó prácticamente blanca. Una repetición controlada con
`lr=1e-5` produjo bandas/mosaico púrpura; la salida `base-only` también fue
inválida. Se conservaron ambos checkpoints y sus hashes en
`docs/experiments/wuerstchen-lora-pilot-2026-08-15.md`.

La GPU no debe usarse para una corrida larga todavía. El diagnóstico posterior
encontró que el validador se apartaba de la receta oficial: 4 pasos uniformes y
256 px, frente a `DEFAULT_STAGE_C_TIMESTEPS`, `float16`, 1024 px y guía 8 para
el checkpoint base. El validador ya está corregido; la siguiente compuerta es
ejecutar `base-only` y obtener una muestra válida. Sólo entonces se vuelve a
evaluar LoRA.
