# Política de licencias de AvatarFace

## Objetivo

AvatarFace sólo incorporará modelos y componentes que permitan el uso previsto,
incluida modificación y redistribución en una aplicación Android, sin
restricciones de campo de uso, royalties ni obligación de publicar derivados.

Esta política es una compuerta técnica y documental; no sustituye revisión
jurídica profesional.

## Licencias admitidas automáticamente

- Apache-2.0.
- MIT.
- BSD-2-Clause.
- BSD-3-Clause.
- CC0-1.0 para activos a los que resulte aplicable.

CC BY puede considerarse para datos si se implementa correctamente la
atribución por activo, pero no se aprueba automáticamente para pesos o código.

## Condiciones no admitidas

- uso exclusivo para investigación;
- prohibición de uso comercial;
- licencias RAIL/OpenRAIL con restricciones de uso;
- términos de uso de modelo no incluidos en la lista aprobada;
- royalties, límites por ingresos o necesidad de licencia comercial adicional;
- obligación de distribuir el producto completo bajo copyleft;
- procedencia o licencia desconocida;
- revisión mutable como `main` sin commit o hash fijado.

## Unidad de auditoría

Un pipeline se divide como mínimo en:

- denoiser, transformer o U-Net;
- encoder de texto;
- tokenizer;
- VAE, VQGAN o autoencoder;
- scheduler o sampler si contiene código redistribuido;
- runtime y librerías nativas;
- pesos de adaptación;
- dataset y activos de calibración.

La licencia declarada en la página principal no se extrapola a dependencias
externas. Cada componente debe registrar URL, revisión, licencia, hash y avisos.

## Estados

- **Aprobado automáticamente:** todos los componentes están fijados y usan una
  licencia de la lista permitida, sin restricciones declaradas.
- **Revisión manual:** el texto parece permisivo, pero faltan componentes,
  revisión, archivo LICENSE o condiciones de acceso.
- **Rechazado:** existe una condición incompatible.

Un resultado automático aprobado es necesario, pero no suficiente para una
descarga o uso en producción. La revisión manual debe comprobar los archivos de
licencia de la revisión exacta.

## Datos y salidas sintéticas

La licencia de los pesos no otorga derechos sobre el dataset original ni
garantiza que cada salida sea utilizable. Para datos sintéticos se registrarán:

- modelo y revisión generadora;
- licencia aplicable;
- prompt y seed;
- filtros de similitud y contenido;
- decisión de aceptación;
- hash del resultado.

No se utilizarán salidas como dataset principal hasta confirmar que la licencia
del maestro permite el flujo de destilación previsto.

## Evidencia y conservación

Por cada componente aprobado se conservarán bajo `models/<id>/` o en su
manifiesto:

- licencia y NOTICE;
- URL canónica;
- commit o revisión;
- SHA-256 de cada archivo;
- fecha de revisión;
- persona o proceso que aprobó;
- obligaciones de distribución.

Los enlaces gated, tokens y aceptaciones de cuenta no se guardan en Git.
