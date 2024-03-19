# README

## INSTALL

```
pip install pdf2gpkg
```

## USAGE

```
usage: pdf2gpkg [-h] [--loglevel {debug,info,warning,error}] [--version] [--curve-density CURVE_DENSITY] [--page-no PAGE_NO] [--ref-name REF_NAME] source_pdf target_dir

positional arguments:
  source_pdf            path to input pdf file
  target_dir            path to output folder (will be created)

options:
  -h, --help            show this help message and exit
  --loglevel {debug,info,warning,error}, -l {debug,info,warning,error}
  --version, -v
  --curve-density CURVE_DENSITY, -d CURVE_DENSITY
                        density of points when converting curves to line segments
  --page-no PAGE_NO, -p PAGE_NO
                        page number in pdf
  --ref-name REF_NAME, -r REF_NAME
                        name of reference map
```

## STEPS

- run `pdf2gpkg` command to initialize output
- in output dir, open `ref.qgz` in QGIS
- edit (and save) layer `ref_src` (and `ref_tgt`) to create 3 pairs of matching coordinates
- run pdf2gpkg again
- `result.gpkg` is your final product, contating a layer each for lines amd text labels
