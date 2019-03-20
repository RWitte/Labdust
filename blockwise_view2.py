from __future__ import division
from builtins import map
import numpy

def blockwise_view( a, blockshape, aslist=False):
    
#    """
#    Return a 2N-D view of the given N-D array, rearranged so each ND block (tile) 
#    of the original array is indexed by its block address using the first N 
#    indexes of the output array.
#    
#    Note: This function is nearly identical to ``skimage.util.view_as_blocks()``, except:
#          - "imperfect" block shapes are permitted (via require_aligned_blocks=False)
#          - only contiguous arrays are accepted.  (This function will NOT silently copy your array.)
#            As a result, the return value is *always* a view of the input.
#    
#    Args:
#        a: The ND array
#        blockshape: The tile shape
#        
#        aslist: If True, return all blocks as a list of ND blocks
#                instead of a 2D array indexed by ND block coordinate.
#        require_aligned_blocks: If True, check to make sure no data is "left over" 
#                                in each row/column/etc. of the output view.
#                                That is, the blockshape must divide evenly into the full array shape.
#                                If False, "leftover" items that cannot be made into complete blocks 
#                                will be discarded from the output view.
# 
#    Here's a 2D example (this function also works for ND):
#    
#    >>> a = numpy.arange(1,21).reshape(4,5)
#    >>> print a
#    [[ 1  2  3  4  5]
#     [ 6  7  8  9 10]
#     [11 12 13 14 15]
#     [16 17 18 19 20]]
#    >>> view = blockwise_view(a, (2,2), False)
#    >>> print view
#    [[[[ 1  2]
#       [ 6  7]]
#    <BLANKLINE>
#      [[ 3  4]
#       [ 8  9]]]
#    <BLANKLINE>
#    <BLANKLINE>
#     [[[11 12]
#       [16 17]]
#    <BLANKLINE>
#      [[13 14]
#       [18 19]]]]
#    Inspired by the 2D example shown here: http://stackoverflow.com/a/8070716/162094
#    """
    blockshape = tuple(blockshape)
    outershape = tuple(numpy.array(a.shape) // blockshape)
    view_shape = outershape + blockshape

    # inner strides: strides within each block (same as original array)
    intra_block_strides = a.strides

    # outer strides: strides from one block to another
    inter_block_strides = tuple(a.strides * numpy.array(blockshape))

    # This is where the magic happens.
    # Generate a view with our new strides (outer+inner).
    view = numpy.lib.stride_tricks.as_strided(a,
                                              shape=view_shape, 
                                              strides=(inter_block_strides+intra_block_strides))

    if aslist:
        return list(map(view.__getitem__, numpy.ndindex(outershape)))
    return view
"""
if __name__ == "__main__":
    import doctest
doctest.testmod()
"""