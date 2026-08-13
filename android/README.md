# Aplicación Android

Este directorio alojará la aplicación Kotlin de referencia cuando la Fase 6
comience. La exportación temprana de la Fase 3 podrá añadir antes un módulo de
benchmark mínimo.

Decisiones vigentes:

- namespace previsto: `com.avatarface.app`;
- ABI primaria: `arm64-v8a`;
- inferencia offline;
- runtime pendiente de benchmark;
- métricas finales obtenidas en dispositivo físico mediante USB y ADB.

No se crea todavía un proyecto Gradle porque el runtime y su matriz de versiones
deben seleccionarse mediante una ADR reproducible.
