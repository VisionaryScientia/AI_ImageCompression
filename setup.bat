@echo off
set PYTHON_VER=3.8
py -%PYTHON_VER% -m ensurepip --upgrade
py -%PYTHON_VER% -m pip install -r requirements.txt
IF %ERRORLEVEL% EQU 0 (
py -%PYTHON_VER% -m pip check tensorflow
)
IF %ERRORLEVEL% EQU 0 (
py -%PYTHON_VER% differentiate.py
py -%PYTHON_VER% image_zlib.py
py -%PYTHON_VER% evaluate_image.py
py -%PYTHON_VER% image_dataset.py
) ELSE (
echo Install needed packages failed
)