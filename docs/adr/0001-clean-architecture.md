# ADR-0001: Clean Architecture y puertos/adaptadores

- Estado: aceptada
- Fecha: 2026-08-13

## Contexto

AvatarFace debe cambiar de modelo, método de cuantización y runtime Android sin
acoplar las reglas del producto a un proveedor.

## Decisión

Separar `domain`, `application`, `infrastructure` y `presentation`. Los casos de
uso dependen de protocolos; la infraestructura los implementa.

## Consecuencias

- Las integraciones requieren adaptadores explícitos.
- Las reglas pueden probarse sin GPU, ADB ni red.
- Se evitarán abstracciones sin al menos un contrato o riesgo real.
