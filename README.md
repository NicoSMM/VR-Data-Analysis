# VR-Data-Analysis

Repositorio para el análisis exploratorio de datos obtenidos en experimentos de realidad virtual, incluyendo señales de eye-tracking, interacción con objetos/AOIs, posición de cabeza y señales fisiológicas registradas durante pruebas piloto.

El objetivo principal del repositorio es organizar scripts de carga, visualización y análisis inicial para estudiar el comportamiento visual y fisiológico de participantes en entornos VR/XR.

---

## Contenido del repositorio

```bash
VR-Data-Analysis/
│
├── data/
│   └── pre-piloto/          # Datos iniciales del pre-piloto  
│   └── piloto/          # Datos iniciales del piloto  
│
├── reports/
│   └── Analysis_details.txt # Detalles o notas del análisis
│
├── src/
│   ├── visualización2D.py   # Visualización 2D inicial de datos de eye-tracking
│   ├── Visualizacion_mejor.py # Visualización 3D interactiva y métricas de mirada
│   ├── test_Z0.py           # Análisis de ECG, Z0, dZ/dt y EDA
│   └── complete_test.py     # Script de prueba
│
├── Z0.png                   # Figura asociada a la señal Z0
├── dz_dt.png                # Figura asociada a dZ/dt
├── .gitignore
└── README.md

```

## Archivos pesados y OneDrive

Este repositorio está dentro de una carpeta sincronizada con OneDrive.

Los archivos pesados del proyecto, como videos, archivos de BioLab y otros registros grandes, se guardan en OneDrive, pero no se suben a GitHub.

GitHub se usa solo para guardar el código, scripts, documentación y archivos livianos.

Por eso, se evita subir automáticamente archivos como:

- videos (`.mp4`, `.avi`, `.mov`, `.mkv`)
- archivos de BioLab (`.mwi`, `.mwx`, `.cfg`)
- archivos comprimidos (`.zip`)
- entornos de Python (`venv/`, `.env`)
- archivos temporales de Python y Jupyter

Estos archivos siguen existiendo en la carpeta de OneDrive https://uccl0-my.sharepoint.com/:f:/g/personal/nicolas_sanmartin_uc_cl/IgCHYbSXOFfUSasDl9HbcKvRATczzJ5sK9nZgRSJrtLXKXY?e=krg8i3 
