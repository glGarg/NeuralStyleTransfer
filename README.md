# Neural Style Transfer :art:
### An implementation of [Neural Style Transfer](https://arxiv.org/abs/1508.06576) (Gatys et al., 2015)
### Usage:
#### Pre-trained VGG model can be downloaded from [here](http://www.vlfeat.org/matconvnet/pretrained/).
#### Run `main.py` with following arguments:
* `-content_image`: path to content image (Required)
* `-style_image`: path to style image (Required)
* `-num_iter`: number of iterations to run on the images (Optional)
* `-v`/`--verbose`: shows the generated image after every n iterations (Optional)

<img src="https://i.imgur.com/lsVvkhE.jpg" height="450" width="400"><img src="https://i.imgur.com/fmQyZFG.jpg=" height="450" width="400">

(Left: style, Right: content, Bottom: result)
