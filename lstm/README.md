```diff  
WIN 10 64 - workign packages versions
numpy==1.16 
pandas==0.22.0
setuptools==41.0.0

try:
    pip install --upgrade --force-reinstall numpy==1.14.5
    pip install --upgrade --force-reinstall pandas==0.22.0
or: 
    pip show #package and delete os global version
```