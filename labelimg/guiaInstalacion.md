# Lanzar labelImg usando `uv`

## si no está instalado `ub`

```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Compilar recursos para este pc

```powershell
    uv run --with pyqt5 --with lxml pyrcc5 resources.qrc -o libs/resources.py
```

## Lanzar labelImg
```powershell
#ubicarse primero en la carpeta del .py
uv run --with pyqt5 --with lxml labelImg.py
```
