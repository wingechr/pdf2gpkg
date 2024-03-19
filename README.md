# README

## INSTALL

```
pip install pdf2gpkg
```

## BASIC USAGE

```
usage: pdf2gpkg [-h] [--loglevel {debug,info,warning,error}] [--version] [--curve-density CURVE_DENSITY] [--page-no PAGE_NO] source_pdf target_dir

positional arguments:
  source_pdf
  target_dir

options:
  -h, --help            show this help message and exit
  --loglevel {debug,info,warning,error}, -l {debug,info,warning,error}
  --version, -v
  --curve-density CURVE_DENSITY, -d CURVE_DENSITY
  --page-no PAGE_NO, -p PAGE_NO
```

## USAGE

- run pdf2gpkg command to initialize output
- in output dir, open `ref.qgz` in QGIS
- edit (and save) layer `ref_src` (and `ref_tgt`) to create 3 pairs of matching coordinates
- run pdf2gpkg again
- `result.gpkg` is your final product, contating a layer each for lines amd text labels
