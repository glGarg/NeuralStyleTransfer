# Neural Style Transfer :art:
### An implementation of [Neural Style Transfer](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.
### Usage:
#### Run `main.py` with following arguments:
* `-content_image`: path to content image (Required)
* `-style_image`: path to style image (Required)
* `-num_iter`: number of iterations to run on the images (Optional)
* `-v`/`--verbose`: shows the generated image after every n iterations (Optional)

![Screenshot](https://i.imgur.com/lsVvkhE.jpg "Example")

(Left: style, Right: content, Bottom: result)

#### ToDo: total variation loss for smoother results, weighted layers cost, more tests :construction:
