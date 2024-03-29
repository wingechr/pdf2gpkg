from setuptools import setup

if __name__ == "__main__":
    with open("README.md", encoding="utf-8") as file:
        long_description = file.read()

    setup(
        name="pdf2gpkg",
        packages=["pdf2gpkg"],
        keywords=[],
        install_requires=["PyMuPDF", "geopandas", "shapely"],
        description="extract and georeference geometry from pdf",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version="0.1.0",
        author="Christian Winger",
        url="https://github.com/wingechr/pdf2gpkg",
        author_email="c@wingechr.de",
        platforms=["any"],
        license="Public Domain",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
            "Operating System :: OS Independent",
        ],
        entry_points={"console_scripts": ["pdf2gpkg = pdf2gpkg.__main__:main"]},
        include_package_data=True,
        package_data={"pdf2gpkg": ["ref/**"]},
    )
