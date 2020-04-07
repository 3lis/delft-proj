"""
#####################################################################################################################

    Delft visiting project
    Alice   2020

    Test functions to evaluate the performance of a model

#####################################################################################################################
"""

# -------------------------------------------------------------------------------------------------------------------
# set global seeds: it must be done at the beginning, in every module
# https://stackoverflow.com/a/60911053/13221459
# https://stackoverflow.com/a/52897289/13221459
SEED = 1
import      os
import      numpy           as np
import      random          as rn
os.environ[ 'PYTHONHASHSEED' ] = str( SEED )
np.random.seed( SEED )
rn.seed( SEED )
# -------------------------------------------------------------------------------------------------------------------

import      cv2
from        PIL         import Image
from        math        import ceil, sqrt
from        print_msg   import print_err, print_wrn



# ===================================================================================================================
#
#   Support functions
#
#   - array_to_image
#   - make_collage
#
# ===================================================================================================================

def array_to_image( pixels ):
    """ -------------------------------------------------------------------------------------------------------------
    Convert np.ndarray to PIL.Image

    NOTE: to convert a PIL.Image.Image to numpy.array, just perform a normal cast:
        $ np.array( img )

    pixels:         [np.ndarray] pixel values

    return:         [PIL.Image.Image]
    ------------------------------------------------------------------------------------------------------------- """
    if len( pixels.shape ) == 4:
        pixels  = pixels[ 0, :, :, : ]                          # remove batch axis

    rgb         = pixels.shape[ -1 ] == 3                       # check whether image is RGB or graylevel 

    if not rgb and len( pixels.shape ) == 3:
        pixels  = pixels[ :, :, 0 ]                             # remove channel in graylevel image

    ptp     = pixels.ptp()                                      # max - min (aka peak-to-peak)

    if ptp > 0:     # if this is false, than the image is totally black and can't be normalized
        pixels  = ( pixels - pixels.min() ) / pixels.ptp()      # normalize between [0..1]
        pixels  = 255 * pixels                                  # normalize between [0..255]

    # NOTE it is necessary to convert into uint8 otherwise Image.fromarray won't work!
    pixels  = np.uint8( pixels )                                # convert to uint8

    if rgb:
        img     = Image.fromarray( pixels, 'RGB' )
    else:
        img     = Image.fromarray( pixels )
        img     = img.convert( 'RGB' )

    return img



def make_collage( imgs, w, h, n_cols=None, n_rows=None, pad_size=5, pad_color="#FFFFFF", save=False ):
    """ -------------------------------------------------------------------------------------------------------------
    Combine a set of images into a collage

    imgs:           [list of PIL.Image.Image] or [np.ndarray] with shape (batch, height, width, channel)
    w:              [int] desired width of single image tile inside the collage
    h:              [int] desired height of single image tile inside the collage
    n_cols:         [int] optional number of columns
    n_rows:         [int] optional number of rows
    pad_size:       [int] pixels between image tiles
    pad_color:      [str] padding color
    save:           [str] filename to save image, False otherwise

    return:         [PIL.Image.Image]
    ------------------------------------------------------------------------------------------------------------- """
    if isinstance( imgs, list ) and isinstance( imgs[ 0 ], np.ndarray ):
        imgs    = np.array( imgs )

    if isinstance( imgs, np.ndarray ):
        imgs    = [ array_to_image( i ) for i in imgs ]

    # from this point, 'imgs' is a list of PIL.Image.Image

    n_imgs  = len( imgs )
    n_cols  = ceil( sqrt( n_imgs ) )    if n_cols is None else n_cols
    n_rows  = ceil( n_imgs / n_cols )   if n_rows is None else n_rows

    width   = n_cols * w + ( n_cols - 1 ) * pad_size
    height  = n_rows * h + ( n_rows - 1 ) * pad_size

    i       = 0
    img     = Image.new( 'RGB', ( width, height ), color=pad_color )
    
    for r in range( n_rows ):
        y   = r * ( h + pad_size )

        for c in range( n_cols ):
            x   = c * ( w + pad_size )
            img.paste( imgs[ i ].resize( ( w, h ) ), ( x, y ) )
            i   += 1
            if i >= n_imgs: break

        if i >= n_imgs: break

    if save is not False: img.save( save )

    return img



# ===================================================================================================================
#
#   Visualization of model prediction
#
#   - pred_from_batch
#
# ===================================================================================================================

def pred_from_batch( model, generator, batch_indx=0, save=False ):
    """ -------------------------------------------------------------------------------------------------------------
    Perform predictions on a batch of data yielded by a generator
    Optionally save collages of images.

    model:          [keras.models.Model] network
    generator:      [DataGenerator] test generator
    batch_indx:     [int] index of batch
    save:           [str] path to folder to save image, False otherwise

    return:         [np.ndarray] batch of predictions
    ------------------------------------------------------------------------------------------------------------- """
    batch_input, batch_target   = generator[ batch_indx ]
    batch_output                = model.predict_on_batch( batch_input )

    def _save( btc, nm ): 
        """ ---------------------------------------------------------------------------------------------------------
        btc:        [np.ndarray] single batch
        nm:         [str] suffix of output file
        --------------------------------------------------------------------------------------------------------- """
        w       = btc.shape[ 2 ]
        w       = w // 10 if w > 200 else w

        h       = btc.shape[ 1 ]
        h       = h // 10 if h > 200 else h

        f       = "batch_{}_{}.png".format( batch_indx, nm )
        f       = os.path.join( save, f )

        make_collage( btc, w, h, save=f )


    if save is not False:
        if not os.path.isdir( save ):
            os.makedirs( save )
        _save( batch_input, 'inp' )
        _save( batch_target, 'trg' )
        _save( batch_output, 'out' )

    return batch_output



# ===================================================================================================================
#
#   Evaluation of model accuracy 
#
#   - match_contour
#   - prec_sample
#   - avgprec_sample
#   - avgprec_batch
#
#   - iou_sample
#   - iou_batch
#
# ===================================================================================================================

def match_contour( c_pred, c_targ, img_size ):
    """ -------------------------------------------------------------------------------------------------------------
    Return a score on the match between two contours. The score is essentially an IoU.

    c_pred:         [list] candidate predicted contour
    c_targ:         [list] target contour
    img_size:       [list] height, width

    return:         [float] score [0..1]
    ------------------------------------------------------------------------------------------------------------- """
    m_targ  = np.zeros( shape=img_size, dtype='uint8' )
    m_pred  = np.zeros( shape=img_size, dtype='uint8' )
    cv2.drawContours( m_targ, [ c_targ ], 0, 255, -1 )      # fill area within the targ contour
    cv2.drawContours( m_pred, [ c_pred ], 0, 255, -1 )      # fill area within the predicted contour
    n_targ  = m_targ[ m_targ==255 ].sum() / 255             # count the pixels in the target area
    n_pred  = m_pred[ m_pred==255 ].sum() / 255             # count the pixels in the predicted area
    n_over  = m_pred[ m_targ==255 ].sum() / 255             # count the pixels in the overlap area

    return n_over / ( n_targ + n_pred )                     # IoU



def prec_sample( c_pred, c_targ, img_size, score_thresh=0.5 ):
    """ -------------------------------------------------------------------------------------------------------------
    Compute the Precision score ( TP/(TP+FP) ) between two lists of contours.

    Build a matrix in which:
        - rows represent contours of target image
        - columns represent contours of predicted image
        - values represent the match score between the two contours

    There is a TP if the match score is above the treshold.
    There is a FP if the same target contour is detected multiple times (in many predicted contours).
    The number of predicted contours is equal to TP+FP.

        Refer to Geiger et al. (2012) "The KITTI Vision Benchmark Suite"

    c_pred:         [list] list with all contours in the predicted image
    c_targ:         [list] list with all contours in the target image
    img_size:       [list] height, width
    score_thresh    [float] consider a match if the score is above this threshold

    return:         [float] score [0..1]
    ------------------------------------------------------------------------------------------------------------- """
    rows    = len( c_targ )                                             # number of target objects
    cols    = len( c_pred )                                             # number of predicted objects

    if rows == 0:
        if cols > 0:    return 0.0                                      # if target is black and prediction is not
        else:           return 1.0                                      # if target and prediction are black
    elif cols == 0:     return 0.0                                      # if prediction is black and target is not

    c_mtrx  = np.zeros( shape=( rows, cols ) )

    # build the matrix of all possible match scores
    for r, t in enumerate( c_targ ):
        for c, p in enumerate( c_pred ):
            c_mtrx[ r, c ]   = match_contour( t, p, img_size)

    hits    = 0                                                         # counter of correct matches
    match   = c_mtrx.max()                                              # find the best match

    # the loop goes from the highest to the lowest match above the threshold
    while match > score_thresh:
        hits    += 1                                                    # one more hit (TP)
        r, c    = np.unravel_index( c_mtrx.argmax(), c_mtrx.shape )     # find the corresponding cell in the matrix
        c_mtrx  = np.delete( c_mtrx, r, axis=0 )                        # delete the corresponding row (target contour)
        c_mtrx  = np.delete( c_mtrx, c, axis=1 )                        # delete the corresponding column (predicted contour)
        if min( c_mtrx.shape ) < 1:                                     # end loop if no more possible matches
            break
        match   = c_mtrx.max()                                          # find the next best match

    return hits / cols



def avgprec_sample( i_pred, i_targ, n_thresh=40 ):
    """ -------------------------------------------------------------------------------------------------------------
    Compute the Average Precision score (AP) of a single sample.

    The same predicted image is binarized with 40 increasing thresholds.
    The Prediction scores are computed over these images (w.r.t. the target image) and then the score is averaged
    in a fancy way to give more importance to the central levels of threshold.

        Refer to    Everingham et al. (2010) "The Pascal Visual Object Classes (VOC) challenge"
                    Simonelli et al. (2019) "Disentangling monocular 3D object detection"

    i_pred:         [np.ndarray] predicted image (graylevel)
    i_targ:         [np.ndarray] target image (binary)
    n_thresh:       [int] number of thresholds from which to compute the average score

    return:         [float] average precision score
    ------------------------------------------------------------------------------------------------------------- """
    i_targ          = np.uint8( 255 * i_targ )                  # convert to uint8 array [0..255] to be used by cv2
    i_targ          = i_targ[ :, :, 0 ]                         # remove channel axis
    i_pred          = i_pred[ :, :, 0 ]
    img_size        = i_targ.shape

    c_targ, _       = cv2.findContours( i_targ, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )    # target contours

    thresholds      = np.linspace( 0, 1., num=n_thresh )                                    # 40 thresholds in [0..1]
    prec            = []                                                                    # keep the 40 scores

    # compute precision score for all the thresholds
    for t in thresholds:
        i_bin               = np.zeros( shape=img_size, dtype='uint8' )
        i_bin[ i_pred>t ]   = 255
        c_pred, _        = cv2.findContours( i_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )    # predicted contours
        prec.append( prec_sample( c_targ, c_pred, img_size ) )

    # At this moment we have a list of 40 scores, each corresponding to a threshold.
    # For all threshold n (in [0..t]), we compute the max among the t-n scores corresponding to the threshold
    # grather than n (that is the t-n values on the right of the list w.r.t. threshold n).
    # The final average is computed on these new 40 scores. This is a way to give more importance to the central
    # (and more fair) thresholds.
    sum_prec    = 0
    for i in range( n_thresh ):
        sum_prec  += max( prec[ i: ] )

    return sum_prec / n_thresh      # final average precision



def avgprec_batch( model, batch_pred, batch_target, n_thresh=40 ):
    """ -------------------------------------------------------------------------------------------------------------
    Compute the Average Precision score (AP) over a batch of data

    model:          [keras.models.Model] network
    batch_input:    [np.ndarray] batch of predicted data
    batch_target:   [np.ndarray] batch of target data
    n_thresh:       [int] number of thresholds from which to compute the average score

    return:         [list of float] average precision scores for each sample of the batch
    ------------------------------------------------------------------------------------------------------------- """
    avg_prec        = []

    for pred, targ in zip( batch_pred, batch_target ):
        score       = avgprec_sample( pred, targ, n_thresh=n_thresh )
        avg_prec.append( score )

    return avg_prec



def iou_sample( pred, targ, threshold=0.5, epsilon=0.001 ):
    """ -------------------------------------------------------------------------------------------------------------
    Compute the IoU score of a single sample

    pred:           [np.ndarray] predicted image (graylevel)
    targ:           [np.ndarray] target image (binary)
    threshold:      [float] used to binarize the predicted segmentation
    epsilon:        [int] minimum fraction of pixels that must be in the union

    return:         [list of float] IoU values for each sample of the batch
    ------------------------------------------------------------------------------------------------------------- """
    pred    = pred > threshold                              # binarize the arrays
    targ    = targ > threshold

    inter   = np.logical_and( targ, pred )
    union   = np.logical_or( targ, pred )

    s_inter = np.sum( inter )                               # number of pixels in the intersection
    s_union = np.sum( union )                               # number of pixels in the union
    p_union = s_union / pred.size * 100                     # fraction of pixels in the union

    # if the size of the union is very small, set the IoU to 1
    # this is the case of an image almost totally black, which is correctly predicted into an almost black image
    if p_union < epsilon: 
        return 1.0
    else:
        return ( s_inter / s_union )



def iou_batch( model, batch_pred, batch_target, threshold=0.5, epsilon=0.001 ):
    """ -------------------------------------------------------------------------------------------------------------
    Compute the IoU score over a batch of data

    model:          [keras.models.Model] network
    batch_input:    [np.ndarray] batch of predicted data
    batch_target:   [np.ndarray] batch of target data
    threshold:      [float] used to binarize the predicted segmentation
    epsilon:        [int] minimum fraction of pixels that must be in the union

    return:         [list of float] IoU values for each sample of the batch
    ------------------------------------------------------------------------------------------------------------- """
    iou             = []

    for pred, targ in zip( batch_pred, batch_target ):
        score       = iou_sample( pred, targ, threshold=threshold, epsilon=epsilon )
        iou.append( score )

    return iou



def eval_dataset( model, generator, iou_trsh=0.5, iou_eps=0.01, ap_trsh=40, save=False ):
    """ -------------------------------------------------------------------------------------------------------------
    Compute the IoU and the Average Precision (AP) scores on batches of data yielded by a generator

    model:          [keras.models.Model] network
    generator:      [DataGenerator] test generator
    threshold:      [float] used to binarize the predicted segmentation
    epsilon:        [int] minimum fraction of pixels that must be in the union
    save:           [str] filename to save txt file, False otherwise

    return:         [np.ndarray], [np.ndarray] IoU and Average Precisions
    ------------------------------------------------------------------------------------------------------------- """
    iou     = []
    ap      = []

    for batch in generator:
        b_in, b_targ    = batch
        b_pred          = model.predict( b_in )

        iou.append( iou_batch( model, b_pred, b_targ, threshold=iou_trsh, epsilon=iou_eps ) )
        ap.append( avgprec_batch( model, b_pred, b_targ, n_thresh=ap_trsh ) )
    
    iou     = np.array( iou )
    ap      = np.array( ap )

    # write results into file
    if save is not False:
        f   = open( save, 'w' )
        f.write( "==== IoU with thresh={} ====\n".format( iou_trsh ) )
        f.write( "MEAN:\t{:.3f}\n".format( np.nanmean( iou ) ) )
        f.write( "VAR:\t{:.3f}\n".format( np.nanvar( iou ) ) )
        f.write( "MIN:\t{:.3f}\n".format( np.nanmin( iou ) ) )
        f.write( "MAX:\t{:.3f}\n".format( np.nanmax( iou ) ) )
        
        f.write( "\n==== Avg Precisions ====\n" )
        f.write( "MEAN:\t{:.3f}\n".format( np.nanmean( ap ) ) )
        f.write( "VAR:\t{:.3f}\n".format( np.nanvar( ap ) ) )
        f.write( "MIN:\t{:.3f}\n".format( np.nanmin( ap ) ) )
        f.write( "MAX:\t{:.3f}\n".format( np.nanmax( ap ) ) )
        f.close()

    return iou, ap
