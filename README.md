System Goal

This system will let u use your computer without touching it


Cómo instalarlo ? 

Entorno Necesario:

Python: Python 3.10 (asegúrate de que el instalador de 64 bits esté instalado, cuando instales Python activa el PATH en el instalador).
pip: Incluido con Python.
Entorno Virtual (venv): Lo crearemos.
Bibliotecas: opencv-python, mediapipe, pyautogui, numpy

Verifica/Instala Python 3.10 (o 3.11):

Abre Git Bash.
Escribe python --version o py --version. Deberías ver la versión 3.10.x o 3.11.x.

Ve a la carpeta que has creado para el proyecto, abre esa dirección de la carpeta en Git Bash. 

# Asegúrate de que 'python' o 'py -3.10' apunte a la versión correcta
python -m venv .venv
Esto creará la carpeta .venv con la estructura necesaria para Windows (incluyendo la subcarpeta Scripts).


Activa el Entorno Virtual (¡El paso clave diferente para Git Bash!):

En Git Bash, para activar un entorno virtual creado en Windows, usas el comando source y apuntas al script activate dentro de la carpeta Scripts, usando barras /:
Bash

source .venv/Scripts/activate


pip install opencv-python mediapipe pyautogui numpy




Ahora estás listo para desarrollar tu proyecto. Recuerda, cada vez que abras Git Bash para trabajar en este proyecto, navega a la carpeta ControlMouseMano y activa el entorno usando:

Bash

source .venv/Scripts/activate





