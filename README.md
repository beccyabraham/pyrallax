# pyrallax
Pair programmed with Evan!

Pyrallax takes in a layered image, created in Procreate or similar,
and creates a parallax animation to add dimension to your artwork. 

## Installation
```
pip install -r requirements.txt
```

## Usage
Prepare your artwork layers by exporting the images to a folder
and numbering the layers, in the order that they're stacked, from bottom up.
Procreate's export PNG layers option works well for this.

## Examples
```
python pyrallax.py examples/kite_layers examples/output/kite.gif --format gif 40
```
![kite animation](examples/output/kite.gif)
```
python pyrallax.py examples/abstract_layers examples/output/abstract.png --format png 40 --x_scales 0 0 .3 .4 .5 .75 1 --y_scales 0 0 .2 .3 .4 .65 .9
```
![abstract animation](examples/output/abstract.png)
```
python pyrallax.py examples/dandelion_layers examples/output/dandelion.png --format png 40 --x_diff .1 --x_freeze 0 1 6 7 8 9 11 --y_diff .1 --y_freeze 0 1 2 3 4 5 6 7 8 9 11 
```
![dandelion animation](examples/output/dandelion.gif)