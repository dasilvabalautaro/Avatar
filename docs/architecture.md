# Arquitectura de AvatarFace

## Objetivo

Mantener las reglas del producto separadas de PyTorch, datasets, Vast.ai, ADB y
el runtime Android. La arquitectura debe permitir comparar implementaciones sin
reescribir los casos de uso.

## Dependencias

```text
presentation ──→ application ──→ domain
                       ↑
infrastructure ────────┘

android app ──→ artefacto exportado + contrato de inferencia
```

El dominio no importa frameworks. La aplicación depende de puertos del dominio.
La infraestructura implementa esos puertos. La presentación compone y ejecuta
los casos de uso.

## Módulos actuales

- `domain.models`: prompts y representación neutral de dispositivos.
- `domain.ports`: contratos para adaptadores.
- `application.inspect_android`: caso de uso de inspección.
- `infrastructure.android.adb_probe`: adaptador ADB.
- `presentation.cli`: entrada de operador.

## Módulos previstos

- dataset y manifiestos;
- selección y descarga auditable de modelos;
- entrenamiento y checkpoints;
- evaluación y regresión visual;
- cuantización y calibración;
- exportación Android;
- almacenamiento de artefactos;
- benchmark en dispositivo.

## Reglas

1. Las rutas externas entran por configuración.
2. Los casos de uso no imprimen ni leen argumentos CLI.
3. Los adaptadores convierten errores de proveedor a resultados explícitos.
4. Los formatos de artefactos incluyen versión de esquema y hashes.
5. Un experimento no sobrescribe silenciosamente otro.
6. La selección del runtime Android se realiza por datos medidos.

## Vertical slice inicial

La CLI `avatar-face status` usa el puerto `AndroidDeviceProbe` y el adaptador
`AdbDeviceProbe`. Este recorrido pequeño valida la dirección de dependencias y
será el patrón para modelos, datasets, cuantización y exportación.
