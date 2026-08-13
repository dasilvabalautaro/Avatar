# Objetivo Android

## Decisiones provisionales

| Propiedad | Valor |
|---|---|
| Plataforma | Android |
| Namespace previsto | `com.avatarface.app` |
| ABI primaria | `arm64-v8a` |
| API mínima provisional | 26 |
| Compile SDK provisional | 35 |
| Lenguaje de aplicación | Kotlin |
| Inferencia | Offline |
| Conexión de pruebas | USB mediante ADB |
| Runtime | Pendiente de benchmark |

La API mínima se revisará después de elegir el runtime. No se reducirá la
compatibilidad sólo por intuición: se documentará qué operador o backend exige
cada límite.

## Entorno detectado el 13 de agosto de 2026

- Android SDK: `/Users/davidsilva/Library/Android/sdk`.
- ADB: versión 37.0.0, disponible en `platform-tools`.
- Platforms instaladas: Android 24 a 36, con algunas extensiones.
- Build Tools: existen versiones hasta 36.0.0.
- NDK: 25.2 y 26.1 instalados.
- Java visible: OpenJDK 25.0.1.
- Gradle global: no encontrado; el proyecto deberá usar Gradle Wrapper.
- Dispositivo ADB: TECNO KM5s autorizado; el serial no se guarda en el
  repositorio.
- Python predeterminado: 3.6.8, incompatible con AvatarFace.
- Python 3.14.2 disponible, útil para las comprobaciones iniciales pero no
  seleccionado para el stack de ML.

## Dispositivo físico inicial

Inventario obtenido por ADB el 13 de agosto de 2026:

| Propiedad | Valor observado |
|---|---|
| Fabricante y modelo | TECNO KM5s |
| Android | 15 |
| API | 35 |
| Parche de seguridad | 2025-11-01 |
| ABI | `arm64-v8a` |
| Hardware | `mt6768` |
| SoC informado | `MT6769` |
| RAM física | 3,833,552 KiB, aproximadamente 3.66 GiB |
| Memoria disponible durante inspección | aproximadamente 1.14 GiB |
| Batería | 96 %, USB conectado, 21 °C |
| Estado térmico Android | 0, sin throttling informado |
| Sensores térmicos expuestos | CPU, GPU, NPU, TPU, SoC, batería y piel |

La presencia de sensores NPU/TPU no demuestra que un delegado concreto pueda
usar esos aceleradores. Esto deberá confirmarse ejecutando el grafo y
registrando el backend real.

Debido a la RAM disponible, el presupuesto del proceso se reduce de 1.5 GiB a
1.0 GiB. El benchmark deberá medir también presión de memoria y cierres del
sistema, no sólo latencia.

## Runtime Python requerido

Python 3.12.14 está instalado y el proyecto utiliza `.venv`. No se reutiliza el
entorno virtual de MythosLight: cada proyecto mantiene sus dependencias,
lockfile y comprobaciones reproducibles.

## Matriz mínima del teléfono de referencia

Cuando se conecte el dispositivo, se registrarán:

- fabricante y modelo;
- versión y parche de Android;
- SoC, CPU, GPU y NPU disponibles;
- RAM total;
- ABI;
- almacenamiento libre;
- estado térmico;
- nivel de batería;
- backends de aceleración utilizables.

Comandos de inventario:

```bash
adb devices -l
adb shell getprop ro.product.manufacturer
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.security_patch
adb shell getprop ro.product.cpu.abi
adb shell cat /proc/meminfo
adb shell dumpsys thermalservice
adb shell dumpsys battery
```

No se ejecutarán comandos sobre un dispositivo hasta verificar su serial. Los
scripts de benchmark aceptarán `--serial` y evitarán seleccionar implícitamente
un teléfono cuando haya más de uno.

## Candidatos de runtime

1. **ExecuTorch:** candidato natural para exportación desde PyTorch.
2. **ONNX Runtime Mobile:** candidato por portabilidad y optimización de grafo.
3. **NNAPI:** se evaluará como backend o delegación disponible, no como garantía
   uniforme entre fabricantes.

La selección se hará con el mismo modelo mínimo y el mismo dispositivo. Se
medirán operadores soportados, tamaño del runtime, tiempo de carga, memoria,
latencia, estabilidad y fallbacks.

## Condiciones de benchmark

- Dispositivo físico conectado por USB.
- Build release o benchmark, nunca debug para métricas finales.
- Mismo conjunto de prompts y seeds.
- Cold start y warm start separados.
- Varias repeticiones con mediana y percentiles.
- Registro de temperatura y batería.
- Detección explícita del backend realmente utilizado.
