# CGMLVQ

Python-Implementierung des Complex Generalized Matrix Learning Vector Quantization Algorithmus.

Für nähere Informationen siehe [Adaptive learning for complex-valued data](https://www.techfak.uni-bielefeld.de/~fschleif/pdf/esann_2012.pdf).

## Voraussetzungen

Es müssen folgende Pakete installieren sein:

* numpy
* scipy

## Verwendung

Um den Algorithmus zu verwenden, importieren Sie den Algorithmus aus dem Modul:

```python
from cgmlvq import CGMLVQ

cgmlvq = CGMLVQ()
cgmlvq.set_params()
cgmlvq.fit( X_train, y_train )

predicted = cgmlvq.predict( X_test )
```

## Autor

Matthias Nunn
