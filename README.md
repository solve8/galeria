# Galería
Una app de galería con etiquetado e identificación de personas en local. Organizar las fotos ya no es necesario.
Usa cálculo de embeddings que se almacenan en una base de datos vectorial y una base de datos sql para los metadatos y etiquetas.

# Decisiones
Se inicializa FAISS usando IndexIDMap, lo que permite que el ID del vector en FAISS sea exactamente el mismo que el ID de la tabla rostros_detectados en SQL. Así, al borrar una foto no se desfasarán y la base de datos no apuntará a caras equivocadas.

- init_dbs.py -> inicializar o bases de datos
- sistema_reconocimiento -> exclusivamente realizar el reconocimiento

- las etiquetas de personas contienen su id

# Ideas futuras
- [] Procesamiento de fotos en paralelo