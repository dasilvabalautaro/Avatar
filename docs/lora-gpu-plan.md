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
