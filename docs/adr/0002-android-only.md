# ADR-0002: Android como única plataforma de esta etapa

- Estado: aceptada
- Fecha: 2026-08-13

## Contexto

El proyecto necesita validar inferencia móvil con recursos limitados. Mantener
dos plataformas antes de fijar el modelo y runtime duplicaría la superficie de
prueba.

## Decisión

Desarrollar, exportar y evaluar exclusivamente para Android. Las métricas de
aceptación se obtendrán en un teléfono físico conectado por USB mediante ADB.

## Consecuencias

- iOS y Core ML quedan fuera del alcance.
- Se prioriza `arm64-v8a`.
- El diseño de operadores se valida contra backends Android.
- Un emulador no sustituye las métricas del dispositivo físico.
