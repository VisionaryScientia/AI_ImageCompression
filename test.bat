@echo off
set PYTHON_VER=3.8
py -%PYTHON_VER% compress.py -c ./images/test/PB080003.JPG ./compressed
IF %ERRORLEVEL% EQU 0 (
py -%PYTHON_VER% compress.py -d ./compressed/PB080003.JPG.bin ./decompressed 
)
IF %ERRORLEVEL% EQU 0 (
start ./decompressed/PB080003.JPG.bin-decompressed.png
)