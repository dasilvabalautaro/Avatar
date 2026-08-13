# AvatarFace

AvatarFace genera rostros de avatares a partir de texto y está diseñado para
inferencia local en Android. Los fundamentos, la auditoría inicial y el spike de
viabilidad móvil están completos; actualmente se desarrolla la Fase 2 de dataset
legal, especializado y auditable.

El plan completo está en
[`docs/plan-proyecto-avatarface.md`](docs/plan-proyecto-avatarface.md).
El estado exacto para retomar otra sesión está en
[`docs/HANDOFF.md`](docs/HANDOFF.md).

## Estado actual

- plataforma de esta etapa: Android;
- pruebas finales: dispositivo físico conectado por USB y controlado con ADB;
- Python requerido: 3.11 o superior; se recomienda 3.12 para el stack de ML;
- runtime provisional: ONNX Runtime CPU con cuantización selectiva;
- baseline sintético Android: 67.5 ms P50 y 78.1 ms P95 en TECNO KM5s;
- smoke dataset: 64 avatares procedimentales auditables;
- plumbing entrenable: loader por manifiesto y microentrenamiento local reanudable verificados;
- modelo generativo entrenable: pendiente de ampliar y aprobar el dataset.

Los inputs de regresión (captions, prompts y seeds) están congelados en
[`configs/regression-fixtures.json`](configs/regression-fixtures.json). Todo
cambio intencional debe actualizar ese archivo y su prueba asociada.

## Licencias de publicación

El código y documentación propios se publican bajo
[Apache-2.0](LICENSE). Las imágenes, captions, atributos y manifiestos del
dataset procedimental propio están dedicados a CC0-1.0; alcance y confirmación
en [`docs/dataset/CC0-DEDICATION.md`](docs/dataset/CC0-DEDICATION.md).
Dependencias, modelos, pesos y activos de terceros conservan sus propias
licencias y pasan por la compuerta de [`docs/license-policy.md`](docs/license-policy.md).

## Desarrollo

Crear un entorno aislado con Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

El entorno local verificado usa Python 3.12.14. Las versiones resueltas de las
herramientas se registran en `requirements-dev.lock`.

Comprobar la base:

```bash
pytest
ruff check .
mypy src
avatar-face status --json
avatar-face audit-candidates --json
avatar-face generate-smoke-dataset --samples 64 --seed 42
avatar-face audit-dataset
```

No se descargan modelos ni datasets durante la instalación base. Datos, pesos,
descargas, secretos y artefactos pesados están excluidos del repositorio.
