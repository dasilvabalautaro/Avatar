# ADR-0004: Compuerta de licencias por componente

- Estado: aceptada
- Fecha: 2026-08-13

## Contexto

Una licencia permisiva del repositorio no garantiza que pesos, encoder de texto,
tokenizer, autoencoder o datos tengan los mismos términos.

## Decisión

Ningún modelo entra al entrenamiento principal sin una matriz que audite todos
sus componentes. Se rechazan términos non-commercial, research-only,
restricciones de campo de uso, royalties u obligaciones incompatibles con la
distribución prevista.

## Consecuencias

- La selección técnica de SANA permanece provisional.
- Apache-2.0 exige conservar avisos y atribuciones.
- Si ningún candidato completo pasa la compuerta, se construirá un estudiante
  con componentes permisivos.
