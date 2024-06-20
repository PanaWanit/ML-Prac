def expand_axis_like(a, b):
    """
    expand axis of a like axis of b
    """
    assert len(a.shape) <= len(b.shape), \
    f"length of a = {len(a.shape)} must less than or equal length of shape b = {len(b.shape)}"
    n_unsqeeze = len(b.shape) - len(a.shape)
    a = a[(..., ) + (None, ) * n_unsqeeze]
    return a