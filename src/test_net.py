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

import      sys
import      cv2
from        PIL         import Image
from        math        import ceil, sqrt

from        extract_nu  import inverse_warp, inverse_grid, forward_warp, forward_grid
from        print_msg   import print_err, print_wrn


WARPED          = None                  # True if the evaluation is performet of warped occupancy grids,
                                        # False in case of standard occupancy grids.
                                        # This is set in main_exec.py

depth_range     = ( 15, 30 )            # thresholds [meter] for the categories CLO/MID/FAR in the occupancy grids

thresh_match    = 0.1                   # minimum score to acknowledge a match
thresh_score    = 0.5                   # minimum score to consider a match in the computation of precision
thresh_eps      = 0.001                 # minimum fraction of pixels required in the union to compute IoU
thresh_avg      = 40                    # number of different thresholds used to compute the average score


# ===================================================================================================================
#
#   Support functions
#
#   - array_to_image
#   - make_collage
#
#   - l2_distance
#   - depth_class
#   - depth_split
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

    imgs:           [list of PIL.Image.Image or np.ndarray] with shape (batch, height, width, channel)
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



def l2_distance( point1, point2 ):
    """ -------------------------------------------------------------------------------------------------------------
    Compute the Euclidean distance between two 2D points.

    point1:         [tuple] ( X, Y )
    point2:         [tuple] ( X, Y )

    return:         [float] distance
    ------------------------------------------------------------------------------------------------------------- """
    p1  = np.array( point1 )
    p2  = np.array( point2 )
    return np.linalg.norm( p2 - p1 )



def depth_class( point ):
    """ -------------------------------------------------------------------------------------------------------------
    Classify the depth range fo the given point in world coordinates.

    point:          [tuple] point in space [m]

    return:         [str] one of CLO/MID/FAR
    ------------------------------------------------------------------------------------------------------------- """
    z   = point[ 1 ]
    if z < depth_range[ 0 ]:    return 'CLO'
    if z < depth_range[ 1 ]:    return 'MID'
    return 'FAR'



def depth_split( image ):
    """ -------------------------------------------------------------------------------------------------------------
    Split an image into three ranges of depth

    image:          [np.ndarray] full image

    return:         [dict] keys:       [str] one of CLO/MID/FAR/ALL
                           values:     [np.ndarray] cropped image
    ------------------------------------------------------------------------------------------------------------- """
    f       = forward_warp if WARPED else forward_grid

    # transform the two ranges of depth from meter to pixel, using the correct transformation function
    y1      = int( f( ( 0, depth_range[ 1 ] ) )[ 1 ] )
    y2      = int( f( ( 0, depth_range[ 0 ] ) )[ 1 ] )

    i       = {
                'FAR':  image[ : y1 ],
                'MID':  image[ y1 : y2 ],
                'CLO':  image[ y2 : ],
                'ALL':  image
    }
    return i



# ===================================================================================================================
#
#   Visualization of model prediction
#
#   - pred_sample
#   - pred_batch
#
# ===================================================================================================================

def pred_sample( model, i_input, i_target, threshold=0.5, save=False ):
    """ -------------------------------------------------------------------------------------------------------------
    Perform prediction on a sample. Optionally save collages of images.

    model:          [keras.models.Model] network
    i_input:        [np.ndarray or PIL.Image.Image or str] input
    i_target:       [np.ndarray or PIL.Image.Image or str] target
    threshold:      [float] binarizing threshold
    save:           [str] path to folder to save image, False otherwise

    return:         [np.ndarray] prediction
    ------------------------------------------------------------------------------------------------------------- """
    if isinstance( i_input, str ):              i_input     = Image.open( i_input )
    if isinstance( i_target, str ):             i_target    = Image.open( i_target )
    if isinstance( i_input, Image.Image ):      i_input     = np.array( i_input )

    if len( i_input.shape ) == 2:
        i_input     = np.expand_dims( i_input, axis=-1 )
    if len( i_input.shape ) == 3:
        i_input     = np.expand_dims( i_input, axis=0 )

    i_output        = model.predict( i_input )
    t_output        = i_output > threshold
    t_output        = t_output.astype( float )

    if save is not False:
        if not os.path.isdir( save ):
            os.makedirs( save )
        i_input     = array_to_image( i_input )
        i_output    = array_to_image( i_output )
        t_output    = array_to_image( t_output )
        i_input.save( os.path.join( save, "input.png" ) )
        i_target.save( os.path.join( save, "target.png" ) )
        i_output.save( os.path.join( save, "output.png" ) )
        t_output.save( os.path.join( save, "output_{:03d}.png".format( int( threshold * 100 ) ) ) )

    return i_output
    
    

def pred_batch( model, generator, batch_indx=0, threshold=0.5, save=False ):
    """ -------------------------------------------------------------------------------------------------------------
    Perform predictions on a batch of data yielded by a generator
    Optionally save collages of images.

    model:          [keras.models.Model] network
    generator:      [DataGenerator] test generator
    batch_indx:     [int] index of batch
    threshold:      [float] binarizing threshold
    save:           [str] path to folder to save image, False otherwise

    return:         [np.ndarray] batch of predictions
    ------------------------------------------------------------------------------------------------------------- """
    batch_input, batch_target   = generator[ batch_indx ]
    batch_output                = model.predict_on_batch( batch_input )

    batch_thresh                = batch_output > threshold
    batch_thresh                = batch_thresh.astype( float )

    def _save( btc, f ): 
        """ ---------------------------------------------------------------------------------------------------------
        btc:        [np.ndarray] single batch
        f:          [str] name of output file (without path)
        --------------------------------------------------------------------------------------------------------- """
        w       = btc.shape[ 2 ]
        w       = w // 10 if w > 200 else w

        h       = btc.shape[ 1 ]
        h       = h // 10 if h > 200 else h

        f       = os.path.join( save, f )
        make_collage( btc, w, h, save=f )


    if save is not False:
        if not os.path.isdir( save ):
            os.makedirs( save )
        _save( batch_input, "batch_{}_inp.png".format( batch_indx ) )
        _save( batch_target, "batch_{}_trg.png".format( batch_indx ) )
        _save( batch_output, "batch_{}_out.png".format( batch_indx ) )
        _save( batch_thresh, "batch_{}_out_{:03d}.png".format( batch_indx, int( threshold * 100 ) ) )

    return batch_output



# ===================================================================================================================
#
#   Functions for computing contours. A contour is the format computed by OpenCV, used to extract object shapes
#   from the "blobs" of a predicted occupancy grid.
#
#   - contour_center
#   - contour_info
#
#   - match_contour
#   - match_contours
#   - match_sample
#
# ===================================================================================================================

def contour_center( contour ):
    """ -------------------------------------------------------------------------------------------------------------
    Return the center of a contour in world coordinates.

    contour:        [numpy.array] with shape ( <n_points>, 1, 2 ) (the format of cv2 contours)

    return:         [tuple] x, z [m]
    ------------------------------------------------------------------------------------------------------------- """
    m       = cv2.moments( contour )

    if m[ 'm00' ] == 0.0:                                   # case of a degenerate contour with just few pixels
        x   = contour[ 0, 0, 0 ]
        y   = contour[ 0, 0, 1 ]
    else:
        x       = m[ 'm10' ] / m[ 'm00' ]                   # center along X
        y       = m[ 'm01' ] / m[ 'm00' ]                   # center along Y

    if WARPED:  return inverse_warp( ( x, y ) )             # apply warped inverse transformation
    return inverse_grid( ( x, y ) )                         # apply linear inverse transformation



def contour_info( contours ):
    """ -------------------------------------------------------------------------------------------------------------
    Scan a list of contours, computing the centers and assigning them to ranges of distance

    contours:       [list] list with contours

    return:         [dict]  depth:  [list of str] one of FAR/MID/CLO
                            center: [list of tuples] centers of contours
    ------------------------------------------------------------------------------------------------------------- """
    if len( contours ) == 0:
        print_err( "There are no contours here. Something went wrong!" )

    res         = { 'depth':[], 'center':[] }

    for i, c in enumerate( contours ):
        center      = contour_center( c )
        depth       = depth_class( center )
        res[ 'center' ].append( center )
        res[ 'depth' ].append( depth )

    return res



def match_contour( c_pred, c_targ, img_size ):
    """ -------------------------------------------------------------------------------------------------------------
    Return a score (IoU) on the match between two contours.

    c_pred:         [numpy.array] with shape ( <n_points>, 1, 2 ) (the format of cv2 contours)
    c_targ:         [numpy.array] with shape ( <n_points>, 1, 2 ) (the format of cv2 contours)
    img_size:       [tuple] height, width

    return:         [float] score [0..1]
    ------------------------------------------------------------------------------------------------------------- """
    m_targ  = np.zeros( shape=img_size, dtype='uint8' )
    m_pred  = np.zeros( shape=img_size, dtype='uint8' )
    m_union = np.zeros( shape=img_size, dtype='uint8' )
    cv2.drawContours( m_targ, [ c_targ ], 0, 255, -1 )      # fill area within the targ contour
    cv2.drawContours( m_pred, [ c_pred ], 0, 255, -1 )      # fill area within the predicted contour
    m_union[ m_targ==255 ]   = 255                          # copy area within the targ contour
    m_union[ m_pred==255 ]   = 255                          # overlap area within the predicted contour
    n_inter = m_pred[ m_targ==255 ].sum() / 255             # count the pixels in the overlap area
    n_union = m_union.sum() / 255                           # count the pixels in the union area

    return n_inter / n_union



def match_contours( c_pred, c_targ, img_size ):
    """ -------------------------------------------------------------------------------------------------------------
    Compute matches between multiple contours, and partition them into ranges of depth.

    For each distance range (FAR/MID/CLO), the function returs a dict containing:
        - 'n_targ':   [int] number of target contours in that range
        - 'n_pred':   [int] number of predicted contours in that range
        - 'n_match':  [int] number of succesful contour matches in that range
        - 'mtc_scr':  [list of float] for each match in that range, its score
        - 'mtc_ctr':  [list of tuples] for each match in that range, the centers of the targ and pred contours

    To assign the matches, build a matrix in which:
        - rows represent all contours of target image
        - columns represent all contours of predicted image
        - values are the score of the possible match between the two contours

    c_pred:         [list] list with all contours in the predicted image
    c_targ:         [list] list with all contours in the target image
    img_size:       [list] height, width

    return:         [dict] keys:       [str] one of CLO/MID/FAR
                           values:     [dict] described above
    ------------------------------------------------------------------------------------------------------------- """
    rows        = len( c_targ )                                         # number of target objects
    cols        = len( c_pred )                                         # number of predicted objects

    res         = {     # initialize the result dict
            'FAR': { 'n_match': 0, 'n_targ': 0, 'n_pred': 0, 'mtc_scr': [], 'mtc_ctr': [] },
            'MID': { 'n_match': 0, 'n_targ': 0, 'n_pred': 0, 'mtc_scr': [], 'mtc_ctr': [] },
            'CLO': { 'n_match': 0, 'n_targ': 0, 'n_pred': 0, 'mtc_scr': [], 'mtc_ctr': [] }
    }

    # stop if there are no target contours...
    if rows == 0:   return res
    # ...otherwise count the number of target contours in each distance range
    i_targ          = contour_info( c_targ )
    for k in res.keys():
        res[ k ][ 'n_targ' ]    = i_targ[ 'depth' ].count( k )

    # stop if there are no predicted contours...
    if cols == 0:   return res                                              
    # ...otherwise count the number of predicted contours in each distance range
    i_pred          = contour_info( c_pred )
    for k in res.keys():
        res[ k ][ 'n_pred' ]    = i_pred[ 'depth' ].count( k )

    # build the matrix of all possible matches between contours
    c_mtrx      = np.zeros( shape=( rows, cols ) )
    for r, t in enumerate( c_targ ):
        for c, p in enumerate( c_pred ):
            c_mtrx[ r, c ]   = match_contour( t, p, img_size )

    mtc_scr     = []
    mtc_idx     = []
    match       = c_mtrx.max()                                          # this variable keeps the current best match score

    # the loop goes from the highest to the lowest match score above the threshold
    while match > thresh_match:
        r, c    = np.unravel_index( c_mtrx.argmax(), c_mtrx.shape )     # the index corresponding to the best match
        mtc_idx.append( ( r, c ) )
        mtc_scr.append( match )
        c_mtrx[ r, : ]  = 0                                             # erase the corresponding row (target contour)
        c_mtrx[ :, c ]  = 0                                             # erase the corresponding column (pred contour)
        match   = c_mtrx.max()                                          # update the current best match score

    # no matches above threshold
    if len( mtc_idx ) == 0:     return res

    for i, tp in enumerate( mtc_idx ):
        t, p    = tp
        t_c     = i_targ[ 'center' ][ t ]                               # target center
        p_c     = i_pred[ 'center' ][ p ]                               # predicted center

        depth_t     = i_targ[ 'depth' ][ t ]
        depth_p     = i_pred[ 'depth' ][ p ]

        # NOTE The distance range of the match is decided by the target.
        # If the target center is in a distance range, while the predicted center is in a different one, the
        # prediction is moved into the target range.
        if not depth_t == depth_p:
            res[ depth_p ][ 'n_pred' ]  -= 1
            res[ depth_t ][ 'n_pred' ]  += 1

        res[ depth_t ][ 'n_match' ]     += 1
        res[ depth_t ][ 'mtc_scr' ].append( mtc_scr[ i ] )
        res[ depth_t ][ 'mtc_ctr' ].append( ( t_c, p_c ) )

    return res



def match_sample( m_pred, m_targ, thresh ):
    """ -------------------------------------------------------------------------------------------------------------
    Compute matches between the prediction and target of a single sample.
    The (graylevel) predicted image is binarized using the given threshold(s).

    m_pred:         [np.ndarray] predicted matrix (graylevel image)
    m_targ:         [np.ndarray] target matrix (binary image)
    thresh:         [float or list of float] binarizing threshold(s)

    return:         [dict or list of dicts] match(es)
    ------------------------------------------------------------------------------------------------------------- """
    matches         = []

    m_targ          = np.uint8( 255 * m_targ )              # convert to uint8 array [0,255] to be used by cv2
    m_targ          = m_targ[ :, :, 0 ]                     # remove channel axis
    m_pred          = m_pred[ :, :, 0 ]                     # remove channel axis
    img_size        = m_targ.shape

    # compute the contours in the target image
    c_targ, _       = cv2.findContours( m_targ, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    thr             = [ thresh, ] if isinstance( thresh, float ) else thresh

    for t in thr:
        # binarize the predicted image using the current threshold
        b_pred              = np.zeros( shape=img_size, dtype='uint8' )
        b_pred[ m_pred>t ]  = 255

        # compute the contours in the predicted image
        c_pred, _           = cv2.findContours( b_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

        # compute the matches between target and predicted contours
        matches.append( match_contours( c_pred, c_targ, img_size ) )

    if isinstance( thresh, float ):
        return matches[ 0 ]

    return matches



# ===================================================================================================================
#
#   Evaluation of model accuracy
#
#   - prec_matches
#   - err_matches
#
#   - prec_sample
#   - err_sample
#   - iou_sample
#
#   - eval_dataset
#
# ===================================================================================================================

def prec_matches( matches ):
    """ -------------------------------------------------------------------------------------------------------------
    For a group of matches, compute the precision score ( TP/(TP+FP) ).

    There is a TP if the match score is above a treshold. The quantity TP+FP is equal to the number of predicted
    contours (that is all the predicted positives, true and false).

        Refer to Geiger et al. (2012) "The KITTI Vision Benchmark Suite"

    matches:        [dict] result of match_contours()

    return:         [dict] keys:       [str] one of CLO/MID/FAR/ALL
                           values:     [float] precision score [0..1]
    ------------------------------------------------------------------------------------------------------------- """
    prec        = { 'FAR': np.NaN, 'MID': np.NaN, 'CLO': np.NaN, 'ALL': np.NaN }    # initialize the result dict
    all_pred    = 0                                             # count total number of predicted contours

    # loop over the distance range
    for dist_range, match_dict in matches.items():

        # for this range, there are no contours in the target and in the pred, so score must be NaN
        if match_dict[ 'n_targ' ] == 0 and match_dict[ 'n_pred' ] == 0:
            continue

        prec[ dist_range ]  = 0                                 # initialize current score to 0
        all_pred            += match_dict[ 'n_pred' ]           # increase total count of predicted contours

        # for this range, there are no matches, so score must be 0
        if match_dict[ 'n_match' ] == 0:
            continue

        # loop over the matches in this distance range
        for s in match_dict[ 'mtc_scr' ]:

            # the match is good enough to be considered a TP
            if s > thresh_score:
                prec[ dist_range ]  += 1                
                prec[ 'ALL' ]       = 1 if prec[ 'ALL' ] is np.NaN else prec[ 'ALL' ] + 1

        # divide the number of TP by the sum TP+FP (which is equal to the total number of predictions)
        prec[ dist_range ]  /= match_dict[ 'n_pred' ]

    prec[ 'ALL' ]       = prec[ 'ALL' ] / all_pred if all_pred > 0 else 0

    return prec



def err_matches( matches ):
    """ -------------------------------------------------------------------------------------------------------------
    For a group of matches, compute the error in the center prediction.

    matches:        [dict] result of match_contours()

    return:         [dict] keys:       [str] one of CLO/MID/FAR/ALL
                           values:     [list of float] errors in meters
    ------------------------------------------------------------------------------------------------------------- """
    err        = { 'FAR': [], 'MID': [], 'CLO': [], 'ALL': [] }

    # loop over the distance range
    for dist_range, match_dict in matches.items():
        if match_dict[ 'n_targ' ] == 0:     continue            # for this range, there are no contours in the target
        if match_dict[ 'n_pred' ] == 0:     continue            # for this range, there are no contours in the prediction
        if match_dict[ 'n_match' ] == 0:    continue            # for this range, there are no matches

        # loop over the matches in this distance range
        for c in match_dict[ 'mtc_ctr' ]:
            t_c, p_c    = c
            e           = l2_distance( t_c, p_c )               # compute error between the centers
            
            err[ dist_range ].append( e )
            err[ 'ALL' ].append( e )

    return err



def prec_sample( pred, targ ):
    """ -------------------------------------------------------------------------------------------------------------
    For a single sample, compute the Average Precision score (AP).

    The function binarizes the predicted (graylevel) image using a number of increasing thresholds,
    producing several predicted binary images.
    It computes precision scores between each of these images and the same constant target image.
    Then the scores are averaged in a fancy way to give more importance to the central levels of threshold.

        Refer to    Everingham et al. (2010) "The Pascal Visual Object Classes (VOC) challenge"
                    Simonelli et al. (2019) "Disentangling monocular 3D object detection"

    pred:           [np.ndarray] predicted image (graylevel image)
    targ:           [np.ndarray] target image (binary image)

    return:         [dict] keys:       [str] one of CLO/MID/FAR/ALL
                           values:     [float] average precision score [0..1]
    ------------------------------------------------------------------------------------------------------------- """
    avg_prec        = { 'FAR': [], 'MID': [], 'CLO': [], 'ALL': [] }

    thresholds      = np.linspace( 0, 1., num=thresh_avg )              # different thresholds in [0..1]
    match_list      = match_sample( pred, targ, thresholds )            # compute a match for each threshold
    prec_list       = [ prec_matches( m ) for m in match_list ]         # compute precision scores for each match

    # each item in 'avg_prec' is a list of 't' precision scores, where 't' is the number of thresholds
    for prec_dict in prec_list:
       for k_dist, v_scores in prec_dict.items(): 
            avg_prec[ k_dist ].append( v_scores )

    # Now, for each element i in [0..t], compute the max among the t-i elements on the right of i in the list.
    # This equals to compute the max precision among the scores corresponding to thresholds greather than i.
    # After repeating this for all t elements of the list, we obtain a new list of t values. We compute
    # the final average on this list.
    # This is a way to give more importance to the central (and more fair) thresholds.

    for k_dist, v_scores in avg_prec.items():
        sum_prec    = 0                                                 # to keep the sum of the max values
        for i in range( thresh_avg ):
            sum_prec  += max( v_scores[ i: ] )

        avg_prec[ k_dist ]  = sum_prec / thresh_avg                     # compute the average

    return avg_prec



def err_sample( pred, targ, threshold=0.5 ):
    """ -------------------------------------------------------------------------------------------------------------
    For a single sample, compute the error in the center prediction

    pred:           [np.ndarray] predicted image (graylevel)
    targ:           [np.ndarray] target image (binary)
    threshold:      [float] binarizing threshold

    return:         [dict] keys:       [str] one of CLO/MID/FAR/ALL
                           values:     [list of float] errors in meters
    ------------------------------------------------------------------------------------------------------------- """
    matches     = match_sample( pred, targ, threshold )
    return err_matches( matches )



def iou_sample( pred, targ, threshold=0.5 ):
    """ -------------------------------------------------------------------------------------------------------------
    For a single sample, compute the IoU score

    pred:           [np.ndarray] predicted image (graylevel)
    targ:           [np.ndarray] target image (binary)
    threshold:      [float] binarizing threshold

    return:         [dict] keys:       [str] one of CLO/MID/FAR/ALL
                           values:     [float] IoU score [0..1]
    ------------------------------------------------------------------------------------------------------------- """
    iou         = { 'FAR': None, 'MID': None, 'CLO': None, 'ALL': None }
    pred_split  = depth_split( pred )
    targ_split  = depth_split( targ )

    def get_iou( pred, targ ):
        """ ---------------------------------------------------------------------------------------------------------
        pred:           [np.ndarray] predicted image (graylevel)
        targ:           [np.ndarray] target image (binary)
        --------------------------------------------------------------------------------------------------------- """
        pred    = pred > threshold                              # binarize the arrays
        targ    = targ > threshold

        inter   = np.logical_and( targ, pred )
        union   = np.logical_or( targ, pred )

        s_inter = np.sum( inter )                               # number of pixels in the intersection
        s_union = np.sum( union )                               # number of pixels in the union
        f_union = s_union / pred.size                           # fraction of pixels in the union

        # if the size of the union is very small, set the IoU to 1
        # this is the case of an almost-black target which is correctly predicted as an almost-black image
        if f_union < thresh_eps:    return 1.0
        else:                       return ( s_inter / s_union )

    for k_dist in iou.keys():
        p               = pred_split[ k_dist ]
        t               = targ_split[ k_dist ]
        iou[ k_dist ]   = get_iou( p, t )

    return iou



def eval_dataset( model, generator, thresh_iou=0.5, thresh_err=0.7, save=False ):
    """ -------------------------------------------------------------------------------------------------------------
    Compute the IoU and the Average Precision (AP) scores on batches of data yielded by a generator

    model:          [keras.models.Model] network
    generator:      [DataGenerator] test generator
    thresh_iou:     [float] binarizing threshold for IoU computation
    thresh_err:     [float] binarizing threshold for center error computation
    save:           [str] filename to save txt file, False otherwise

    return:         [dict,dict] raw results and statistics
    ------------------------------------------------------------------------------------------------------------- """
    dist_keys   = ( 'ALL', 'CLO', 'MID', 'FAR' )                            # ordered
    stat_keys   = ( 'AVG', 'VAR', 'MIN', 'MAX' )                            # ordered

    res         = {
            'prec': { 'FAR': [], 'MID': [], 'CLO': [], 'ALL': [] },
            'err':  { 'FAR': [], 'MID': [], 'CLO': [], 'ALL': [] },
            'iou':  { 'FAR': [], 'MID': [], 'CLO': [], 'ALL': [] }
    }
    stat        = {
            'prec': { 'FAR': {}, 'MID': {}, 'CLO': {}, 'ALL': {} },
            'err':  { 'FAR': {}, 'MID': {}, 'CLO': {}, 'ALL': {} },
            'iou':  { 'FAR': {}, 'MID': {}, 'CLO': {}, 'ALL': {} }
    }

    for i in range( len( generator ) ):
        batch           = generator[ i ]
        b_in, b_targ    = batch
        b_pred          = model.predict_on_batch( b_in )

        for pred, targ in zip( b_pred, b_targ ):
            prec    = prec_sample( pred, targ )
            iou     = iou_sample( pred, targ, threshold=thresh_iou )
            err     = err_sample( pred, targ, threshold=thresh_err )

            # collect all metrics
            for k in dist_keys:
                res[ 'prec' ][ k ].append( prec[ k ] )
                res[ 'iou' ][ k ].append( iou[ k ] )
                res[ 'err' ][ k ]   += err[ k ]             # err[ k ] is a list, so don't append but concatenate

    # compute statistic measures
    for kd in dist_keys:
        for km in res.keys():
            a   = np.array( res[ km ][ kd ] )
            stat[ km ][ kd ][ 'AVG' ]   = np.nanmean( a )
            stat[ km ][ kd ][ 'VAR' ]   = np.nanvar( a )
            stat[ km ][ kd ][ 'MIN' ]   = np.nanmin( a )
            stat[ km ][ kd ][ 'MAX' ]   = np.nanmax( a )

    def print_stat( f, km, title ):
        """ ---------------------------------------------------------------------------------------------------------
        f:          [_io.TextIOWrapper]
        km:         [str] one of prec/err/iou
        title:      [str] title of the table
        --------------------------------------------------------------------------------------------------------- """
        f.write( '+' + 49 * '-' + '+\n' )
        f.write( "|{:^49}|\n".format( title ) )
        f.write( '+' + 49 * '-' + '+\n' )
        f.write( "|{:^9}|{:^9}|{:^9}|{:^9}|{:^9}|\n".format( '', *stat_keys ) )
        for kd in dist_keys:
            f.write( 5 * ( '+' + 9 * '-' ) + '+\n' )
            f.write( "|{:^9}".format( kd ) )                                        # distance range
            for ks in stat[ km ][ k ].keys():
                f.write( "|{:^9.3f}".format( stat[ km ][ kd ][ ks ] ) )             # metric score
            f.write( '|\n' )
        f.write( '+' + 49 * '-' + '+\n\n' )

    # write results into file
    if save is not False:
        f   = open( save, 'w' )
        print_stat( f, 'iou', "IoU ( t={} )".format( thresh_iou ) )
        print_stat( f, 'prec', "Average Precision score" )
        print_stat( f, 'err', "Error in center prediction ( t={} )".format( thresh_err ) )
        f.close()

    return res, stat
