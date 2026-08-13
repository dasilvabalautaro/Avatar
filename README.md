# AvatarFace

AvatarFace genera rostros de avatares a partir de texto y está diseñado para
inferencia local en Android. El proyecto se encuentra en la Fase 0: fundamentos,
arquitectura y definición verificable del dispositivo objetivo.

El plan completo está en
[`docs/plan-proyecto-avatarface.md`](docs/plan-proyecto-avatarface.md).

## Estado actual

- plataforma de esta etapa: Android;
- pruebas finales: dispositivo físico conectado por USB y controlado con ADB;
- Python requerido: 3.11 o superior; se recomienda 3.12 para el stack de ML;
- runtime Android: pendiente de benchmark entre ExecuTorch, ONNX Runtime Mobile
  y NNAPI;
- modelo base: pendiente de la compuerta de licencias y del benchmark técnico.

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
```

No se descargan modelos ni datasets durante la instalación base. Datos, pesos,
descargas, secretos y artefactos pesados están excluidos del repositorio.
