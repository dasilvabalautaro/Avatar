# Plan del proyecto AvatarFace

## 1. Resumen ejecutivo

AvatarFace será un sistema de Inteligencia Artificial Generativa capaz de crear
rostros de avatares a partir de texto. El modelo se especializará en una tarea
acotada —generación de rostros de avatar— y se optimizará para ejecutar la
inferencia completamente en dispositivos Android.

En esta etapa el proyecto no contempla iOS. Cuando sea necesario validar el
runtime móvil, se conectará un dispositivo Android físico por USB a la máquina
de desarrollo y se realizarán pruebas automatizadas mediante ADB. Los emuladores
podrán utilizarse para comprobar la aplicación, pero no se aceptarán como fuente
de métricas finales de latencia, memoria, temperatura o consumo energético.

El proyecto incluirá:

- un modelo especializado en rostros de avatares;
- entrenamiento reproducible en GPU alquilada en Vast.ai;
- inferencia de referencia en Python;
- cuantización INT8 y evaluación de INT4/INT8 mixta;
- exportación e integración en una aplicación Android de referencia;
- pruebas sobre un teléfono Android real conectado por USB;
- evaluación de calidad, rendimiento, sesgos, seguridad y licencias;
- documentación técnica y operativa completa.

## 2. Alcance

### 2.1 Incluido en esta etapa

- Entrada de texto y generación de un rostro de avatar.
- Ejecución offline en Android.
- Resolución mínima de 256 × 256 píxeles.
- Resolución de 512 × 512 como objetivo condicionado al rendimiento real.
- Una aplicación Android de referencia para probar el modelo.
- Integración preferente con Kotlin y una capa nativa sólo si el runtime lo
  requiere.
- Evaluación de ExecuTorch, ONNX Runtime Mobile y NNAPI para seleccionar un
  único runtime de producción.
- Uso de CPU, GPU o NPU disponibles en Android mediante el backend seleccionado.
- Pruebas funcionales y de rendimiento mediante ADB y USB.
- Entrenamiento, destilación, poda, cuantización y exportación.
- Trazabilidad completa de datos, código, pesos y dependencias.

### 2.2 Fuera de alcance

- iOS, Core ML y Apple Neural Engine.
- Aplicaciones web o servicios cloud de inferencia para producción.
- Video, animación facial, lip sync o generación 3D.
- Generación de cuerpo completo.
- Entrenamiento directamente en el teléfono.
- Compatibilidad garantizada con todos los dispositivos Android desde la
  primera versión.

## 3. Definición de éxito

Los objetivos siguientes son iniciales. Se confirmarán después de identificar
el teléfono de prueba y ejecutar el primer benchmark del runtime.

| Métrica | Objetivo inicial |
|---|---:|
| Plataforma | Android únicamente |
| Ejecución | 100 % local y offline |
| Resolución obligatoria | 256 × 256 |
| Resolución deseable | 512 × 512 |
| Pasos de inferencia | 1–4 |
| Tamaño total del modelo | ≤250 MB objetivo; ≤400 MB máximo inicial |
| Memoria máxima de la aplicación | ≤1.0 GB en el dispositivo de referencia |
| Latencia en Android de referencia | ≤5 s a 256 × 256 |
| Precisión principal | INT8 |
| Precisión avanzada | INT4/INT8 mixta |
| Degradación tras cuantización | ≤5 % en métricas acordadas frente a FP16 |
| Estabilidad | ≥99 % de prompts válidos sin error |
| Reproducibilidad | resultado equivalente para modelo, prompt y seed fijos |

Además de las métricas agregadas, la aceptación requerirá inspección visual de
ojos, dientes, cabello, accesorios, simetría, bordes y coherencia del estilo.

## 4. Principios y restricciones

1. **Android primero:** toda decisión de arquitectura se justificará contra el
   hardware y los operadores disponibles en Android.
2. **Licencias permisivas:** se admitirán componentes que permitan uso
   comercial, modificación y redistribución, sin restricciones de campo de uso,
   royalties ni obligación de publicar derivados. Obligaciones normales como
   avisos y atribución de Apache-2.0 sí deberán cumplirse.
3. **Auditoría por componente:** la licencia del repositorio no basta; se
   auditarán por separado código, pesos, encoder de texto, tokenizer,
   autoencoder, datasets y runtime.
4. **Mobile desde el diseño:** el tamaño, número de pasos, memoria y operadores
   soportados se tratarán como restricciones del modelo, no como una conversión
   tardía.
5. **Reproducibilidad:** toda corrida conservará configuración, commit, hashes,
   seed, versiones, hardware, duración y costo.
6. **Limpieza:** las descargas reconstruibles se eliminarán después de ser
   verificadas, procesadas y respaldadas según el flujo definido.
7. **Optimización permanente:** cada fase concluirá con recomendaciones y una
   revisión de la frontera entre calidad, costo, tamaño, memoria y latencia.

## 5. Selección preliminar del modelo

### 5.1 Candidato de investigación

SANA 0.6B es un candidato técnico inicial por su arquitectura eficiente, su
variante de 512 píxeles y sus caminos de cuantización de 4 y 8 bits. No se
aprueba automáticamente como base de producción: aunque el repositorio y los
pesos declaran Apache-2.0, el pipeline utiliza Gemma 2 como encoder de texto y
su tarjeta remite a términos adicionales.

SANA-Sprint también se evaluará técnicamente por su inferencia de un paso, pero
estará sujeto a la misma compuerta de licencias y a validación de operadores en
Android.

### 5.2 Estrategias que se compararán

**Ruta A — adaptar un modelo existente**

- Seleccionar un modelo eficiente sólo después de aprobar todos sus
  componentes y dependencias.
- Especializarlo con LoRA o fine-tuning parcial.
- Destilarlo a pocos pasos y cuantizarlo.
- Sustituir componentes restrictivos o demasiado grandes cuando sea viable.

**Ruta B — construir un estudiante propio permisivo**

- Encoder de texto objetivo: 20–60 millones de parámetros.
- Denoiser eficiente objetivo: 100–200 millones de parámetros.
- Autoencoder objetivo: 15–40 millones de parámetros.
- Total objetivo: 150–300 millones de parámetros antes de cuantización.
- Arquitectura restringida a operadores compatibles con el runtime Android.

La decisión se basará en una frontera de Pareto que compare licencia, calidad,
tamaño, RAM, latencia, compatibilidad Android y costo de entrenamiento.

## 6. Arquitectura de software

Se aplicará Clean Architecture, inspirada en la separación utilizada por el
proyecto MythosLight, sin copiar sus componentes específicos de edición de
imágenes.

```text
AvatarFace/
├── android/                         # aplicación Android de referencia
├── configs/                         # perfiles reproducibles
├── docs/                            # documentación y ADR
├── scripts/                         # Vast.ai, Drive, exportación y QA
├── src/avatar_face/
│   ├── domain/                      # entidades, contratos y políticas
│   ├── application/                 # casos de uso
│   ├── infrastructure/              # PyTorch, datasets, Vast, runtimes
│   ├── presentation/                # CLI
│   └── config/                      # configuración tipada
├── tests/
├── transfer/                        # hashes e instrucciones, no secretos
└── pyproject.toml
```

### 6.1 Capas

- **Domain:** entidades, value objects, contratos y reglas independientes de
  PyTorch, Android, Vast.ai y proveedores externos.
- **Application:** preparación de datos, entrenamiento, evaluación,
  cuantización, exportación y generación.
- **Infrastructure:** implementaciones para PyTorch, Hugging Face, Drive,
  Vast.ai, SafeTensors, ONNX Runtime y ExecuTorch.
- **Presentation:** comandos CLI y adaptadores de entrada/salida.
- **Android:** aplicación de referencia y benchmark instrumental.

### 6.2 Puertos principales

- `DatasetRepository`
- `ModelRepository`
- `Trainer`
- `CheckpointStore`
- `Quantizer`
- `ModelExporter`
- `InferenceRuntime`
- `Evaluator`
- `ArtifactStore`
- `ExperimentTracker`
- `LicenseAuditor`
- `AndroidBenchmarkRunner`

### 6.3 Patrones

- Repository para datasets, modelos y artefactos.
- Strategy para entrenamiento, cuantización, exportación y runtime.
- Adapter para Hugging Face, Drive, Vast.ai, ADB y runtimes Android.
- Factory para construir el modelo o backend indicado por configuración.
- Builder para datasets y paquetes de transferencia reproducibles.
- Command/Use Case para mantener la lógica fuera de la CLI.
- Observer para métricas, logs, checkpoints y alertas.
- Specification para reglas de licencia y criterios de aceptación.

## 7. Plan de ejecución

### Fase 0 — Fundamentos y alcance Android

**Duración:** 1 semana.

Tareas:

- Inicializar el repositorio, `pyproject.toml`, estructura `src/` y pruebas.
- Definir el estilo de los avatares: estilizado, semirrealista o fotográfico.
- Identificar modelo, SoC, RAM, versión de Android y ABI del teléfono físico.
- Definir API mínima de Android y arquitecturas soportadas; inicialmente
  `arm64-v8a`.
- Instalar Android SDK Platform Tools y validar ADB.
- Definir idiomas y gramática de prompts.
- Crear taxonomía de atributos del rostro.
- Crear ADR, registro de riesgos y presupuesto de GPU.

Entregables:

- `docs/product-requirements.md`
- `docs/android-target.md`
- `docs/risk-register.md`
- `docs/adr/`
- matriz de aceptación medible.

Compuerta: no aprobar objetivos definitivos de latencia o memoria sin conocer el
dispositivo físico de referencia.

### Fase 1 — Auditoría de modelos y licencias

**Duración:** 1–2 semanas.

Tareas:

- Comparar modelos eficientes y arquitecturas móviles reproducibles.
- Auditar código, pesos, encoder de texto, tokenizer, autoencoder y runtime.
- Rechazar licencias non-commercial, research-only o con restricciones de campo
  de uso incompatibles.
- Descargar revisiones inmutables y calcular SHA-256.
- Crear un manifiesto por modelo.
- Ejecutar un benchmark FP16 inicial.
- Verificar exportabilidad de operadores antes de entrenar.
- Elegir Ruta A, Ruta B o una transición explícita entre ambas.

Entregables:

- `docs/model-selection.md`
- `docs/license-policy.md`
- `docs/model-license-matrix.md`
- manifiestos de modelo con revisión y hashes.

Compuerta: ningún peso podrá entrar al pipeline principal sin aprobación de la
matriz completa.

### Fase 2 — Dataset legal, especializado y auditable

**Duración:** 2–4 semanas.

Tareas:

- Priorizar avatares sintéticos, activos CC0, CC BY o producidos para el
  proyecto.
- Evitar por defecto datasets de rostros con restricciones comerciales o de
  redistribución.
- Conservar licencia, procedencia, autor y autorización por muestra.
- Separar entrenamiento, validación, test y regresión visual.
- Detectar duplicados y similitud no deseada con identidades reales.
- Balancear tonos de piel, edades aparentes, expresiones y estilos.
- Crear captions naturales y atributos estructurados.
- Construir un smoke dataset pequeño para desarrollo local.

Cada muestra deberá registrar como mínimo:

```json
{
  "id": "...",
  "image": "...",
  "caption": "...",
  "attributes": {},
  "source": "...",
  "creator": "...",
  "license": "...",
  "license_url": "...",
  "consent_or_release": "...",
  "sha256": "...",
  "split": "train"
}
```

Entregables:

- datasheet del dataset;
- manifiesto versionado;
- reporte de duplicados y contaminación;
- matriz de licencias;
- scripts reproducibles de preparación.

Compuerta: 100 % de las muestras deberá tener procedencia y licencia aceptadas.

### Fase 3 — Baseline reproducible

**Duración:** 2 semanas.

Tareas:

- Implementar configuración tipada y CLI.
- Implementar descarga por revisión y hash.
- Generar baseline FP16.
- Congelar un conjunto de prompts y seeds de regresión.
- Medir calidad, VRAM, RAM y tiempo.
- Ejecutar microentrenamiento local.
- Ejecutar smoke training en Vast.ai.
- Verificar checkpoints y reanudación.
- Probar una exportación temprana y cargarla en Android.

Comandos previstos:

```text
avatar-face download-model
avatar-face check-model
avatar-face prepare-dataset
avatar-face train
avatar-face evaluate
avatar-face quantize
avatar-face export-android
avatar-face preflight-vast
```

### Fase 4 — Especialización y compresión estructural

**Duración:** 2–4 semanas.

Orden de experimentación:

1. LoRA para validar dataset, prompts y estética.
2. Fine-tuning parcial si LoRA resulta insuficiente.
3. Destilación hacia un estudiante pequeño.
4. Destilación de classifier-free guidance y reducción a 1–4 pasos.
5. Poda estructurada de bloques o canales de baja contribución.
6. Fine-tuning de recuperación.
7. Destilación o sustitución del encoder de texto, si corresponde.

Técnicas:

- BF16/FP16 y AMP;
- gradient checkpointing y accumulation;
- EMA;
- atención eficiente;
- checkpoints reanudables;
- seeds fijos;
- early stopping según calidad y costo;
- optimizer de 8 bits si demuestra estabilidad.

### Fase 5 — Cuantización orientada a Android

**Duración:** 2–3 semanas.

Variantes mínimas:

- FP16 como referencia;
- INT8 PTQ como primer baseline móvil;
- INT8 QAT como candidato estable;
- INT4/INT8 mixta, conservando en mayor precisión las capas sensibles.

Proceso:

1. Crear conjunto de calibración representativo.
2. Analizar sensibilidad por capa.
3. Proteger embeddings, atención cruzada, normalizaciones y capas de entrada y
   salida cuando sea necesario.
4. Aplicar PTQ y medir calidad y alineamiento.
5. Aplicar QAT si PTQ supera el límite de degradación.
6. Medir la ganancia real en Android, no sólo el tamaño del archivo.
7. Crear la frontera de Pareto de calidad, tamaño, RAM, energía y latencia.

Compuerta: INT4 sólo será candidato de producción si mejora el rendimiento del
backend real sin degradación visual inaceptable.

### Fase 6 — Aplicación y runtime Android

**Duración:** 2–4 semanas.

Tareas:

- Crear una aplicación Android mínima en Kotlin.
- Implementar entrada de prompt, seed, ejecución, cancelación y guardado.
- Evaluar ExecuTorch, ONNX Runtime Mobile y NNAPI.
- Elegir un runtime de producción mediante una ADR.
- Exportar por componentes cuando reduzca la memoria máxima.
- Aplicar optimización de grafo y constant folding.
- Empaquetar tokenizer, configuración, licencias y versión del modelo.
- Evitar copias innecesarias entre CPU, GPU y NPU.
- Implementar warm-up controlado y reporte de progreso.
- Instrumentar latencia por etapa, RAM, errores y backend utilizado.

Entregables:

- APK de prueba;
- modelo Android firmado por hash;
- benchmark instrumental;
- guía de integración;
- matriz de compatibilidad del dispositivo probado.

### Fase 7 — Pruebas Android por USB

**Duración:** continua desde la primera exportación; 1 semana de estabilización.

Preparación del dispositivo:

1. Activar opciones de desarrollador y depuración USB.
2. Conectar el teléfono por USB.
3. Autorizar la huella RSA de la máquina.
4. Confirmar que ADB reconoce exactamente el dispositivo esperado.

Comprobaciones iniciales previstas:

```bash
adb devices -l
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.product.cpu.abi
adb shell dumpsys meminfo com.avatarface.app
```

Flujo automatizado:

1. Compilar APK de benchmark.
2. Instalarlo mediante ADB.
3. Copiar, instalar o verificar el modelo según la estrategia de empaquetado.
4. Ejecutar prompts y seeds congelados.
5. Capturar resultados, logs y tiempos.
6. Medir cold start y warm start.
7. Consultar memoria con `dumpsys meminfo`.
8. Realizar varias generaciones consecutivas para observar throttling térmico.
9. Extraer imágenes y reporte JSON a `artifacts/android/`.
10. Comparar hashes, métricas y regresión visual.

Reglas de prueba:

- El benchmark final se ejecutará sin el teléfono cargando si la medición de
  energía o temperatura puede verse alterada.
- Se cerrarán aplicaciones de fondo no esenciales.
- Se documentarán nivel de batería, temperatura inicial, versión de Android,
  modo energético y backend.
- Las mediciones se repetirán y reportarán con mediana y percentiles, no con una
  sola ejecución.
- ADB no se expondrá por red; se utilizará USB salvo necesidad documentada.

### Fase 8 — Evaluación integral

**Duración:** continua; 1–2 semanas para la aceptación.

Calidad del modelo:

- alineamiento texto-imagen;
- DINO o LPIPS para regresión;
- FID/KID como señales agregadas;
- diversidad por prompt y seed;
- exactitud de atributos;
- artefactos faciales;
- similitud no deseada con personas reales;
- sesgos por tono de piel, género aparente y edad;
- evaluación humana ciega.

Rendimiento Android:

- tamaño de APK y modelo;
- tiempo de carga;
- cold start y warm start;
- latencia total y por componente;
- memoria máxima;
- estabilidad tras generaciones consecutivas;
- temperatura y throttling;
- consumo energético cuando la instrumentación disponible sea suficiente;
- fallos por operador o fallback inesperado a CPU.

### Fase 9 — Entrega y mantenimiento

**Duración:** 1 semana inicial y mantenimiento continuo.

Entregables:

- model card;
- datasheet;
- guía de entrenamiento en Vast.ai;
- guía de cuantización;
- guía de exportación e integración Android;
- guía de pruebas por USB/ADB;
- SBOM y matriz de licencias;
- changelog del modelo;
- política de versiones y rollback;
- informe final del dispositivo probado.

## 8. Entrenamiento en Vast.ai

El proyecto reutilizará las buenas prácticas operativas de MythosLight:

- plantilla PyTorch con versión fija;
- validación de CUDA antes de instalar dependencias;
- preflight de GPU, VRAM, disco, commit y hashes;
- smoke test antes de una corrida pagada;
- AMP y checkpoints reanudables;
- logs y manifiestos por experimento;
- exportación de artefactos antes de destruir la instancia.

El preflight de AvatarFace verificará además:

- dataset y licencia-manifest completos;
- revisión exacta del modelo;
- operadores y versiones necesarias para la exportación Android;
- espacio para checkpoints y paquetes de salida;
- ejecución real de una operación CUDA;
- estado limpio o registrado del código utilizado.

## 9. Transferencia local → Drive → Vast.ai

Se respetará obligatoriamente el flujo indicado para el proyecto.

### 9.1 Preparación local

1. Descargar fuentes en un directorio temporal controlado.
2. Verificar tamaño, hash, licencia y estructura.
3. Procesar y materializar únicamente los datos necesarios.
4. Eliminar los archivos descargados que ya sean reconstruibles.
5. Crear paquetes `.tar` independientes, sin gzip para contenido ya comprimido.
6. Calcular un único `SHA256SUMS`.
7. Validar cada paquete con `tar -tf`.
8. Subir los paquetes y el manifiesto a Google Drive.
9. Mantener temporalmente el `.tar` local sólo hasta verificar la restauración
   y el smoke test en Vast.ai.
10. Eliminar el `.tar` local después de esa verificación si Drive queda como
    respaldo autorizado.

### 9.2 Restauración en Vast.ai

Para cada paquete, de uno en uno:

1. Descargar desde Drive con reanudación.
2. Verificar que el nombre figure en `SHA256SUMS`.
3. Verificar SHA-256.
4. Auditar la estructura del tar y prevenir path traversal.
5. Extraer exclusivamente bajo `/workspace/AvatarFace`.
6. Verificar archivos, conteos y manifiestos.
7. Eliminar inmediatamente el `.tar` descargado.
8. Continuar con el paquete siguiente.

### 9.3 Salida de Vast.ai

- Empaquetar checkpoints, métricas, muestras y manifiestos.
- Calcular hashes antes de subir.
- Subir a Drive.
- Descargar o inspeccionar una copia para verificar integridad.
- Eliminar paquetes temporales de Vast.ai.
- Destruir la instancia sólo después de validar el respaldo.

Los enlaces de Drive, credenciales y secretos nunca se guardarán en Git.

## 10. Documentación

La estructura documental prevista es:

```text
docs/
├── plan-proyecto-avatarface.md
├── product-requirements.md
├── architecture.md
├── android-target.md
├── model-selection.md
├── license-policy.md
├── dataset-datasheet.md
├── training.md
├── vast-ai-setup.md
├── quantization.md
├── android-export.md
├── android-usb-testing.md
├── evaluation.md
├── security-and-safety.md
├── optimization-backlog.md
├── experiments/
└── adr/
```

También se mantendrán docstrings públicas, ayuda de CLI, ejemplos
reproducibles, diagramas, model card por versión y documentación de
experimentos fallidos.

## 11. Calidad de software

Stack inicial previsto:

- Python 3.11 o 3.12;
- PyTorch;
- Diffusers y Transformers sólo en infraestructura, si son necesarios;
- SafeTensors;
- Pydantic Settings;
- Typer;
- pytest;
- Ruff;
- mypy en modo estricto;
- ONNX y ONNX Runtime para la ruta que corresponda;
- ExecuTorch para la ruta que corresponda;
- Kotlin, Gradle y Android SDK;
- ADB para instalación, ejecución y diagnóstico.

Pipeline de calidad:

```text
lint → type-check → unit tests → integration tests → model smoke test
→ export test → APK test → USB device regression → performance gate
```

Datos, pesos, secretos, descargas y artefactos pesados quedarán fuera del
control de versiones.

## 12. Optimización continua

Al terminar cada fase se actualizará `docs/optimization-backlog.md` con:

- sugerencia;
- evidencia;
- impacto esperado;
- esfuerzo y costo;
- riesgo;
- prioridad;
- resultado medido cuando se implemente.

Se revisarán permanentemente:

- costo de GPU por mejora obtenida;
- bloques que puedan congelarse, podarse o cuantizarse;
- datos redundantes o de baja contribución;
- cuellos de botella de I/O y preprocessing;
- tamaño y latencia de cada componente;
- operadores que causen fallback en Android;
- calidad de las métricas frente a evaluación humana;
- experimentos que deban detenerse anticipadamente;
- cambios que muevan realmente la frontera de Pareto.

## 13. Riesgos principales y mitigación

| Riesgo | Mitigación |
|---|---|
| El modelo base contiene términos incompatibles | Auditoría por componente y Ruta B permisiva |
| El encoder de texto domina el tamaño | Distilación o sustitución por encoder pequeño |
| INT4 reduce calidad o no acelera Android | INT8 como baseline y precisión mixta por sensibilidad |
| Operadores no soportados | Exportación temprana antes del entrenamiento principal |
| Fallback silencioso a CPU | Instrumentación del backend y benchmark por etapa |
| Dataset con licencia o consentimiento insuficiente | Manifiesto obligatorio y compuerta del 100 % |
| Parecido con personas reales | Datos sintéticos, deduplicación y evaluación de similitud |
| Throttling térmico | Pruebas repetidas en dispositivo físico |
| Pérdida de una corrida Vast.ai | Checkpoints reanudables y respaldo externo incremental |
| Descargas ocupan disco innecesariamente | Extracción por paquete y limpieza verificada inmediata |

## 14. Cronograma orientativo

Estimación para una persona especializada en ML con trabajo adicional de
integración Android:

| Fase | Duración estimada |
|---|---:|
| Fundamentos y objetivo Android | 1 semana |
| Modelos y licencias | 1–2 semanas |
| Dataset inicial | 2–4 semanas |
| Baseline reproducible | 2 semanas |
| Especialización y destilación | 2–4 semanas |
| Cuantización | 2–3 semanas |
| Aplicación y runtime Android | 2–4 semanas |
| Estabilización y aceptación | 1–2 semanas |
| Total orientativo | 13–22 semanas |

Hitos sugeridos:

1. **Semana 2:** modelo y componentes con licencia decidida.
2. **Semana 4–6:** baseline FP16 evaluado.
3. **Semana 7–9:** estudiante especializado de pocos pasos.
4. **Semana 9–12:** primera exportación INT8 cargada en Android.
5. **Semana 11–15:** APK probado por USB en dispositivo físico.
6. **Semana 13–22:** optimización, INT4 mixta si aporta valor y entrega.

## 15. Próximo paso recomendado

La primera ejecución debe implementar la Fase 0 y producir un vertical slice
pequeño: estructura Python limpia, CLI, un modelo mínimo de prueba, exportación
temprana, aplicación Android que lo cargue y un script ADB que ejecute un caso
instrumentado. Este recorrido reducirá pronto el mayor riesgo técnico del
proyecto: descubrir demasiado tarde que una arquitectura o cuantización no se
ejecuta eficientemente en Android.
