class settings():
    """
    Modify this settings for processing
    """
    batch_size = 128
    inpaint_size = 8
    input_size = 24
    # max processing size, if an input image has bigger dims it is resized for faster processing
    max_dim = 640
    # grayscale images only
    channels = 1
    # use low-variance block mask
    zero_diff = 0
    # loss can be 'binary_crossentropy', "mean_squared_error" or empty string for inference mode
    loss = 'mean_squared_error'
    learning_rate = 0.001