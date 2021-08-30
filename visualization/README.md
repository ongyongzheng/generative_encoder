# The Visualization Class

This folder contains the visualization modules (using Visdom) for GE.

Implemented Visualizers:

- Image Visualizer (handles both 1 and 3 channel images)

## Required Components of the Visualization Class

1. The ```publishTensors``` Function

```python
def publishTensors(data, out_size_image, caption="", window_token=None, env="main", nrow=16):
    """
    publish the tensors to visdom images

    Inputs:
        data           [Tensor: (batch_size, num_channels, height, width)] - tensor array containing image (raw output) to be displayed.
        out_size_image [Tuple: (height, width)]                            - expected height and width of the visualization window.
    """
    global vis

    # perform post processing for the image to be published
    outdata = resizeTensor(data, out_size_image)

    return vis.images(outdata, opts=dict(caption=caption), win=window_token, env=env, nrow=nrow)
```

2. The ```saveTensor``` Function

```python
def saveTensor(data, out_size_image, path):
    """
    save the tensors to path

    Inputs:
        data           [Tensor: (batch_size, num_channels, height, width)] - tensor array containing image (raw output) to be displayed.
        out_size_image [Tuple: (height, width)]                            - expected height and width of the visualization window.
        path           [String]                                            - path to save the image to.
    """

    # perform post processing for the image to be saved
    outdata = resizeTensor(data, out_size_image)

    vutils.save_image(outdata, path)
```

3. The ```publishLoss``` Function

```python
def publishLoss(data, name="", window_tokens=None, env="main"):
    """
    publish the loss to visdom images

    Inputs:
        data [Dict] - dictionary containing key (title) and value (list of loss) to be displayed.
    """

    if window_tokens is None:
        window_tokens = {key: None for key in data}

    for key, plot in data.items():

        # skip metadata not needed to be printed
        if key in ("scale", "iter"):
            continue

        nItems = len(plot)
        inputY = np.array([plot[x] for x in range(nItems) if plot[x] is not None])
        inputX = np.array([data["iter"][x] for x in range(nItems) if plot[x] is not None])

        opts = {'title': key + (' scale %d loss over time' % data["scale"]),
                'legend': [key], 'xlabel': 'iteration', 'ylabel': 'loss'}

        window_tokens[key] = vis.line(X=inputX, Y=inputY, opts=opts,
                                      win=window_tokens[key], env=env)

    return window_tokens
```