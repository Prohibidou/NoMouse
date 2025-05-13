System Goal
This system lets you use your computer without touching it, using just your hand.

How to install it?

Required Environment:

Python: Python 3.10 (make sure you install the 64-bit installer; when installing Python, enable “Add to PATH”).

pip: Included with Python.

Virtual Environment (venv): We will create one.

Libraries: opencv-python, mediapipe, pyautogui, numpy

Verify/Install Python 3.10 (or 3.11):

Open Git Bash.

Run:

python --version
# or
py --version
You should see version 3.10.x or 3.11.x.

Navigate to your project folder in Git Bash.
Make sure that python (or py -3.10) points to the correct version.

Create the virtual environment:

python -m venv .venv
This creates a folder named .venv with the necessary structure for Windows (including the Scripts subfolder).

Activate the virtual environment (key step for Git Bash!):
In Git Bash, use the source command and point to the activate script inside Scripts, using forward slashes:

source .venv/Scripts/activate
Install the required libraries:


pip install opencv-python mediapipe pyautogui numpy
You are now ready to develop your project.
Reminder: Each time you open Git Bash to work on this project, navigate to the ControlMouseMano folder and activate the virtual environment with:


source .venv/Scripts/activate


