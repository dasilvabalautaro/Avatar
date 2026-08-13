# Aplicación Android de viabilidad

Aplicación Kotlin mínima para ejecutar el grafo ONNX sintético. No es la UI del
producto y el modelo contiene pesos aleatorios.

Decisiones vigentes:

- namespace previsto: `com.avatarface.app`;
- ABI primaria: `arm64-v8a`;
- inferencia offline;
- primer runtime: ONNX Runtime Android 1.23.2;
- métricas finales obtenidas en dispositivo físico mediante USB y ADB.

## Preparación

```bash
../.venv/bin/avatar-face export-feasibility --profile micro --overwrite
```

El directorio `artifacts/feasibility` se incorpora como fuente de assets durante
el build, pero sus pesos generados permanecen fuera de Git.

## Build

```bash
JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home \
  ./gradlew --no-daemon :app:assembleDebug
```

## Ejecución

La ejecución automatizada exige el serial para impedir que una prueba alcance el
dispositivo equivocado:

```bash
../.venv/bin/avatar-face benchmark-android \
  --serial 14254155BM000874 \
  --runs 7
```

Se puede repetir sólo un backend con `--backend cpu` o `--backend nnapi`. La
herramienta instala el APK, ejecuta cada backend y guarda los JSON bajo
`artifacts/android/`.

Para el artefacto cuantizado:

```bash
../.venv/bin/avatar-face benchmark-android \
  --serial 14254155BM000874 \
  --model-asset avatarface-feasibility-micro-int8.onnx \
  --runs 7
```

Los componentes pueden medirse sustituyendo `--model-asset` por uno de:

- `avatarface-feasibility-bridge-encoder.onnx`;
- `avatarface-feasibility-bridge-denoiser.onnx`;
- `avatarface-feasibility-bridge-decoder.onnx`;
- sus variantes `-int8-preprocessed.onnx` disponibles.

Para guardar el perfil por operador de ONNX Runtime se añade
`--profile-operators`. El JSON se extrae automáticamente junto al resultado del
benchmark bajo `artifacts/android/`.

El pipeline selectivo persistente se ejecuta como un modelo virtual que conecta
tres assets:

```bash
../.venv/bin/avatar-face benchmark-android \
  --serial 14254155BM000874 \
  --model-asset avatarface-feasibility-bridge-selective.onnx \
  --backend cpu \
  --runs 30
```

Manualmente, el benchmark puede iniciarse con extras `backend=cpu|nnapi` y
`runs=N`. El resultado queda en `files/benchmark-result.json` y puede
recuperarse con `run-as` en el build debug.
