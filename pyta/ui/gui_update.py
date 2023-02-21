import PyQt6.uic

fpath = 'gui.ui'

with open('gui.py', 'w') as file:
    PyQt6.uic.compileUi(fpath, file)
