"""
#####################################################################################################################

    Extraction of ground truth from the nuScenes dataset

    Alice   2020

#####################################################################################################################

    Implemented so far:
        - location map (graylevel mask of all 2D boxes in a frame)
        - attention map (RGB mask of all 2D boxes in a frame)
        - occupancy grid
        - warped occupancy grid
        - odometry

    Requirement: installation of the nuscenes package
        $ pip install nuscenes-devkit

    The dataset is organized in a series of dictionaries, as interfaces to a relational database.
    Each entity is associated with a token, a string used as unique identifier. 
    The dataset contains two main kinds of information:
        - samples:  (also called "keyframes") selection of annotated data, at 2 fps
        - sweeps:   all available image frames (some without annotation), at 10 fps

    The ground truths are available for "samples" only, but the temporal sampling is too coarse.
    I can obtain GT values for "sweeps" as well by interpolating the GT values from "samples".

    Odometry is taken from the CAN bus data, in the "Pose" messages, sampled at 50Hz.
    The synchronization with the "sweeps" frame is ensured by checking the relative timestamps.

#####################################################################################################################
"""

import  os
import  re
import  sys
import  shutil
import  numpy                           as np

from    PIL                             import Image, ImageDraw, ImageFilter
from    pyquaternion.quaternion         import Quaternion
from    shapely.geometry                import MultiPoint, box

# allow external programs to import modules from this file without NuScenes installation
no_nuscene      = False
try:
    from    nuscenes.nuscenes               import NuScenes
except ImportError:
    no_nuscene  = True
if not no_nuscene:
    from    nuscenes.utils.geometry_utils   import view_points
    from    nuscenes.can_bus.can_bus_api    import NuScenesCanBus


# ===================================================================================================================
#   CONSTANTs
# ===================================================================================================================

verbose         = True 

DO_EXEC         = False                                 # parse all nuScenes sequences
DO_ATTN         = True                                  # compute and save location and attention maps
DO_GRID         = True                                  # compute and save occupancy grids
DO_MASK         = False                                 # overlay a triangular mask on the warped occupancy grid
DO_ODOM         = True                                  # retrieve and save odometry
DO_FLST         = True                                  # write the list of all frame filenames for a scene
REPLACE         = False                                 # replace entirely the directory tree

img_size        = ( 1600, 900 )                         # size (in pixels) of nuScenes images
grid_size       = (  128, 128 )                         # size (in pixels) of the linear occupancy grid
warp_size       = (  128, 128 )                         # size (in pixels) of the warped occupancy grid
warp_mask       = (   40,  20 )                         # triangular mask for the lower part of the warped occupancy grid
grid_scale      = 0.5                                   # size (in meters) of a pixel in the linear occupancy grid
warp_fact       = 2.0                                   # project_warp factor in the warped occupancy grid
grid_foreground = 3.5                                   # offset in depth of the visible grid foreground (in meters)

# NOTE The value 0.5 of grid_scaleis used by Kim et al. (2017),
# "Probabilistic vehicle trajectory prediction over occupancy grid map via recurrent neural network"

# root dir of nuScenes dataset
dir_root        = '/archive/dataset/nuscenes'
dir_root        = '/Users/alice/Works/D4C/autoencoder_project/delft/dataset/nuScenes'
dir_root        = '/workspace/dataset/nuScenes'
# root dir of original dataset (must be an absolute path)
dir_orig        = os.path.join( dir_root, 'orig' )
# root dir of new extracted GT
dir_new         = os.path.join( dir_root, 'data' )

nuscenes_seqs   = [ 407 ]                               # TEMP
nuscenes_seqs   = list( range( 850 ) )                  # indices of all the sequences in nuScenes

nuscenes_ver    = 'v1.0-trainval'                       # original directory with .json files
nuscenes_cam    = 'CAM_FRONT'                           # original nuScenes camera used here
nuscenes_obj    = [                                     # original classes of objects of interest
        'vehicle.bus.bendy',
        'vehicle.bus.rigid',
        'vehicle.car',
        'vehicle.emergency.ambulance',
        'vehicle.emergency.police',
        'vehicle.trailer',
        'vehicle.truck'
]

fname_img       = "filenames.txt"                       # file saving the original names of the frames in the sequence
fname_odom      = "odometry_stat.txt"                   # file saving some odometry statistics for the sequence

str_frm_loc     = "loc_{:04d}-{:03d}.jpg"               # string format for location map
str_frm_att     = "att_{:04d}-{:03d}.jpg"               # string format for attention map
str_frm_occ     = "occ_{:04d}-{:03d}.png"               # string format for linear occupancy grid
str_frm_wrp     = "wrp_{:04d}-{:03d}.png"               # string format for warped occupancy grid
str_frm_odm     = "odm_{:04d}-{:03d}.npy"               # string format for odometry


# ===================================================================================================================
#   GLOBALs
# ===================================================================================================================

# global instances of the NuScenes interfaces (NOTE it takes some time, but this is the first necessary operation)
#nu              = NuScenes( version=nuscenes_ver, dataroot=dir_orig, verbose=verbose )      # NuScenes database
#nc              = NuScenesCanBus( dataroot=dir_orig )                                       # NuScenes CAN bus
can_msg         = None                                  # CAN bus messages of an entire sequence


# -----------------------------------------------------------------------------------------------------------
# derived constants to be used for computing log-occupancy grid map
warp_offs       = np.log( warp_fact )
warp_zmax       = grid_scale * warp_size[ 1 ]
warp_scale      = warp_zmax / ( np.log( warp_zmax + warp_fact ) - warp_offs )
warp_foreground = warp_scale * ( np.log( grid_foreground + warp_fact ) - warp_offs )



def init_nu():
    """ -------------------------------------------------------------------------------------------------------------
    Initialize nuScene

    return:         [tuple] NuScenes instance, NuScenesCanBus instance
    ------------------------------------------------------------------------------------------------------------- """
    nu              = NuScenes( version=nuscenes_ver, dataroot=dir_orig, verbose=verbose )      # NuScenes database
    nc              = NuScenesCanBus( dataroot=dir_orig )                                       # NuScenes CAN bus
    return nu, nc



# ===================================================================================================================
#   
#   Check how many black images are contained in the sequences
#
#   - check_occ
#   - check_descr
#   - check_seq
#
# ===================================================================================================================

def check_occ( img, thresh=0.01 ):
    """ -------------------------------------------------------------------------------------------------------------
    Check the number of white pixels in an occupancy grid

    img:            [str] pathname of the occupancy grid to check
    thresh:         [float] minimum fraction of white pixles in the occupancy grid

    return:         [int] 0=full black, 1=below occ_fraction, 2=above occ_fraction
    ------------------------------------------------------------------------------------------------------------- """
    occ     = np.array( Image.open( img ) )
    n       = occ.shape[ 0 ] * occ.shape[ 1 ]
    whites  = ( occ > 0 ).sum()

    if not whites:              return 0        # totally black
    if whites / n < thresh:     return 1        # number of white pixels below the threshold
    return 2                                    # number of white pixels above the threshold



def check_descr( desrc ):
    """ -------------------------------------------------------------------------------------------------------------
    Check the description of a nuScene sequence and return a list of bool for 3 conditions

    desrc:          [str] description returned by nu.scene[ index ][ 'description' ]

    return:         [list of bool]  1. ego-car is stationary
                                    2. night sequence
                                    3. rain sequence
    ------------------------------------------------------------------------------------------------------------- """
    cond1   = bool( re.match( r'^wait at', desrc, re.IGNORECASE ) )         # starts with...
    cond2   = bool( re.match( r'^waiting', desrc, re.IGNORECASE ) )         # starts with...
    cond3   = bool( re.search( 'night', desrc, re.IGNORECASE ) )        # contains...
    cond4   = bool( re.search( 'rain', desrc, re.IGNORECASE ) )         # contains...

    return ( cond1 or cond2 ), cond3, cond4



def check_seq( thresh=0.01 ):
    """ -------------------------------------------------------------------------------------------------------------
    Check all nuScenes sequences for number of black images, stationary videos and night/rain conditions

    thresh:         [float] minimum fraction of white pixles in the occupancy grid

    return:         [dict] key = name of sequence, value =  1. fraction of images with enough white pixels
                                                            2. conditions on the description of the sequence
    ------------------------------------------------------------------------------------------------------------- """
    dict_seq        = dict()
    list_seq        = [ f for f in sorted( os.listdir( dir_new ) ) if f.startswith( "scene-" ) ]
    list_dscr       = [ s[ 'description' ] for s in nu.scene ]

    for i, seq in enumerate( list_seq ):
        n_tot       = 0
        n_ok        = 0

        list_occ    = [ f for f in sorted( os.listdir( os.path.join( dir_new, seq ) ) ) if f.startswith( "occ_" ) ]
        for o in list_occ:
            if check_occ( os.path.join( dir_new, seq, o ), thresh=thresh ) == 2:
                n_ok    += 1
            n_tot   += 1

        dict_seq[ seq ]     = [ np.round( n_ok / n_tot, decimals=3 ), check_descr( list_dscr[ i ] ) ]

    np.save( os.path.join( dir_root, 'black_list.npy' ), dict_seq )
    return dict_seq
    


# ===================================================================================================================
#
#   Trasformations between occupancy grids and real world coordinates
#   The formulas of the inverse transformations are computed in Wolfram Mathematica (see 'inv_trans.m')
#
#   - inverse_grid
#   - forward_grid
#
#   - inverse_warp
#   - forward_warp
#
# ===================================================================================================================

def inverse_grid( point ):
    """ -------------------------------------------------------------------------------------------------------------
    Transform pixel coordinates of a standard occupancy grid into real world coordinates.

    point:          [tuple] coordinates (X,Y) of point in the image [pixel]

    return:         [tuple] coordinates (x,z) of the point in real space [m]
    ------------------------------------------------------------------------------------------------------------- """
    x       = 0.5 * grid_scale * ( 2 * point[ 0 ] - grid_size[ 0 ] )
    z       = grid_foreground + grid_scale * ( grid_size[ 1 ] - point[ 1 ] )

    return x, z



def forward_grid( point ):
    """ -------------------------------------------------------------------------------------------------------------
    Transform real world coordinates into pixel coordinates of a standard occupancy grid.

    point:          [tuple] coordinates (x,z) of the point in real space [m]

    return:         [tuple] coordinates (X,Y) of point in the image [pixel]
    ------------------------------------------------------------------------------------------------------------- """
    x_, z       = point
    y_off       = grid_foreground / grid_scale
    x_offset    = grid_size[ 0 ] / 2
    y_offset    = grid_size[ 1 ] + y_off
    x           = x_offset + x / grid_scale
    y           = y_offset - z / grid_scale
    return x, y



def inverse_warp( point ):
    """ -------------------------------------------------------------------------------------------------------------
    Transform pixel coordinates of an warped occupancy grid into real world coordinates.

    point:          [tuple] coordinates (X,Y) of the pixel in the warped grid

    return:         [tuple] coordinates (x,z) of the point in real space [m]
    ------------------------------------------------------------------------------------------------------------- """
    y_off   = warp_foreground / grid_scale

    e       = -1 + point[ 1 ] / warp_size[ 1 ]
    b       = warp_fact / ( warp_fact + grid_scale * warp_size[ 1 ] )
    z       = np.power( b, e )
    z       *= ( grid_foreground + warp_fact )
    z       -= warp_fact

    x       = z * ( point[ 0 ] - 0.5 * warp_size[ 0 ] )
    x       *= np.log( warp_fact + grid_scale * warp_size[ 0 ] ) - warp_offs
    x       /= warp_size[ 0 ] * ( np.log( warp_fact + z ) - warp_offs )

    return x, z



def forward_warp( point ):
    """ -------------------------------------------------------------------------------------------------------------
    Transform real world coordinates into pixel coordinates of a warped occupancy grid.

    point:          [tuple] coordinates (x,z) of the point in real space [m]

    return:         [tuple] coordinates (X,Y) of the pixel in the warped grid
    ------------------------------------------------------------------------------------------------------------- """
    x_, z_      = point
    z           = warp_scale * ( np.log( z_ + warp_fact ) - warp_offs )
    x           = x_ * z / z_
    y_off       = warp_foreground / grid_scale
    x_offset    = warp_size[ 0 ] / 2
    y_offset    = warp_size[ 1 ] + y_off
    x           = x_offset + x / grid_scale
    y           = y_offset - z / grid_scale
    return x, y



# ===================================================================================================================
#
#   Projections of 3D bounding boxes into camera view and bird's eye view
#
#   - project_warp
#   - project_bev
#   - project_camera
#   - compute_boxes
#
# ===================================================================================================================

def project_warp( points ):
    """ -------------------------------------------------------------------------------------------------------------
    Project world-space coordinates in the warped occupancy grid, using the following formula:
        z'  = z_max * ( log( z + c ) - log( c ) ) / ( log( z_max + c ) - log( c ) )
        x'  = x z'/z
    where c corresponds to warp_fact

    points:         [np.array] coordinates (x,z) of point to be projected

    return:         [np.array] coordinates with same format of input
    ------------------------------------------------------------------------------------------------------------- """
    pts     = points.transpose()                # just for easier operations

    def _warp( point ):
        x_, z_  = point
        z       = warp_scale * ( np.log( z_ + warp_fact ) - warp_offs )
        x       = x_ * z / z_

        return np.array( ( x, z ) )

    # check for valid logarithm value
    if pts[ 1 ].min() + warp_fact <= 0:
        return None

    return _warp( pts ).transpose()             # revert to original format



def project_bev( corners ):
    """ -------------------------------------------------------------------------------------------------------------
    Project a 3D bounding box into bird's-eye view.

    The input format is the one returned by nuscenes.utils.data_classes.Box.corners(), that is a np.array
    with shape 3x8 (8 points in 3D) where:
        - the first 4 corners belong to the face closer to the "camera" in clkw-ord starting from the bottom right
        - the other 4 corners belong to the face farther to the "camera" in clkw-ord starting from the bottom right
    Therefore, the corners in clkw-ord forming the projected 2D box are the 1st, 2nd, 6th, 5th of the array
    
    corners:        [np.array] the 8 corners of the 3D box

    return:         [np.array] coordinates of the 2D box (in clockwise order) in bird's-eye view
    ------------------------------------------------------------------------------------------------------------- """
    coo     = corners[ ::2 ].transpose()    # discard Y (heigth) and transpose
    indx    = np.array( [ 0, 1, 5, 4 ] )    # the indices of the 4 useflul corners, in clockwise order
    return coo[ indx ]
    


def project_camera( corners, cam_model ):
    """ -------------------------------------------------------------------------------------------------------------
    Project a 3D bounding box into camera (image) space (the origin of the camera space is top left corner).

    This function is taken from post_process_coords() in
    nuscenes-devkit/python-sdk/nuscenes/scripts/export_2d_annotations_as_json.py
    
    corners:        [np.array] the 8 corners of the 3D box
    cam_model:      ???

    return:         [tuple of int] ??? INT OR FLOAT
                    coordinates of the 2D box, None if the object is outside the camera frame
    ------------------------------------------------------------------------------------------------------------- """
    front       = np.argwhere( corners[ 2, :] > 0 )         # check which corners are in front of the camera
    corners     = corners[ :, front.flatten() ]             # and take those only

    corners_p   = view_points( corners, cam_model, True )   # project the 3D corners in camera space
    corners_p   = corners_p.T[ :, : 2 ]                     # take only X and Y in camera space

    poly        = MultiPoint( corners_p.tolist() ).convex_hull
    img         = box( 0, 0, img_size[ 0 ], img_size[ 1 ] )
    if not poly.intersects( img ):
        return None                                         # return None if the projection is out of the camera frame

    inters  = poly.intersection( img )
    coords  = np.array( [ c for c in inters.exterior.coords ] )
    min_x   = min( coords[ :, 0 ] )
    min_y   = min( coords[ :, 1 ] )
    max_x   = max( coords[ :, 0 ] )
    max_y   = max( coords[ :, 1 ] )

    return min_x, min_y, max_x, max_y
    


def compute_boxes( cam, ego, b ):
    """ -------------------------------------------------------------------------------------------------------------
    Compute from a 3D nuscenes.utils.data_classes.Box object, the 2D projections in camera and bird's-eye views

    This function is taken from get_2d_boxes() in
    nuscenes-devkit/python-sdk/nuscenes/scripts/export_2d_annotations_as_json.py

    cam:            [tuple] ??? <cam_trans>, <cam_rot>, <cam_model>
    ego:            [tuple] ??? <ego_trans>, <ego_rot>
    b:              [nuscenes.utils.data_classes.Box] the 3D bounding box object

    return:         [tuple of ???] coordinates of the 2D projection in camera view
                    [np.array] coordinates of the 2D projection in bird's-eye view
    ------------------------------------------------------------------------------------------------------------- """
    cam_trans, cam_rot, cam_model   = cam
    ego_trans, ego_rot              = ego

    b.translate( -ego_trans )                           # translate into ego pose frame
    b.rotate( ego_rot.inverse )                         # and rotate into ego pose frame
    b.translate( -cam_trans )                           # translate into front camera frame
    b.rotate( cam_rot.inverse )                         # and rotate into front camera frame

    corners = b.corners()                               # the 8 corners of the 3D box
    p_bev   = project_bev( corners )                    # the projection in bird's eye view
    p_cam   = project_camera( corners, cam_model )      # the projection in camera view

    if p_cam is None:
        return None
    return p_cam, p_bev
    

# TODO review of comments arrived HERE

# ===================================================================================================================
#
#   - fill_grid
#
#   - stat_odo
#   - get_odo
#
#   - get_frame_data
#   - track_frame
#   - track_frames
#   - track_nuscenes
#   - summary
#
# ===================================================================================================================

def fill_grid( grid, points, log_grid=False ):
    """ -------------------------------------------------------------------------------------------------------------
    Draw a single polygon in the occupancy grid

    grid:           [PIL.ImageDraw.ImageDraw] occupancy grid
    points:         [np.array] coordinates (x,z) of points
    log_grid:       [bool] True if the grid has logarithmic scale
    ------------------------------------------------------------------------------------------------------------- """
    if log_grid:
        size    = warp_size
        y_off   = warp_foreground / grid_scale
    else:
        size    = grid_size
        y_off   = grid_foreground / grid_scale
    x_offset    = size[ 0 ] / 2
    y_offset    = size[ 1 ] + y_off
    corners     = points / grid_scale                               # scale from [m] to [pixel]
    corners     *= np.array( [ 1, -1 ] )                            # invert the longitudinal/vertical dimension
    corners     += np.array( [ x_offset, y_offset ] )               # offset so that the origin is the top left of the image

    xy          = [ tuple( p ) for p in corners.astype( int ) ]     # polygon() accepts only tuples

    # NOTE polygon() is able to handle the case some of the corners fall out of the image, so no need to check for that
    grid.polygon( xy, fill='white', outline=None )



def mask_grid( grid, log_grid=True ):
    """ -------------------------------------------------------------------------------------------------------------
    Draw a double triangular mask for the non visible part in a grid

    grid:           [PIL.ImageDraw.ImageDraw] occupancy grid
    log_grid:       [bool] True if the grid has logarithmic scale
    ------------------------------------------------------------------------------------------------------------- """
    if log_grid:
        size    = warp_size
    else:
        size    = grid_size

    xm          = size[ 0 ] -1
    ym          = size[ 1 ] -1

    x           = warp_mask[ 0 ]
    y           = size[ 1 ] - warp_mask[ 1 ]
    xy          = [ ( 0, y ), ( x, ym ), ( 0, ym ) ]
    grid.polygon( xy, fill='black', outline=None )

    x           = xm - x
    xy          = [ ( x, ym ), ( xm, y ), ( xm, ym ) ]
    grid.polygon( xy, fill='black', outline=None )


def stat_odo():
    """ -------------------------------------------------------------------------------------------------------------
    produce a statistics of the odomatry

    return:         [tuple] <cam_trans> <cam_rot> <cam_model> <ego_trans> <ego_rot>

    ------------------------------------------------------------------------------------------------------------- """
    agv_sp      = 0.0
    agv_st      = 0.0
    min_sp      = 100.0
    min_st      = 10.0
    max_sp      = 0.0
    max_st      = -10.0
    n           = len( can_msg )

    for m in can_msg:
        speed       = m[ 'vel' ][ 0 ]                   # forward speed
        steer       = m[ 'rotation_rate' ][ -1 ]        # rotation rate
        agv_sp      += speed
        agv_st      += steer
        if speed > max_sp:  max_sp = speed
        if speed < min_sp:  min_sp = speed
        if steer > max_st:  max_st = steer
        if steer < min_st:  min_st = steer

    agv_sp      /= n
    agv_st      /= n
    return { 'speed' : ( min_sp, agv_sp, max_sp ), 'steer' : ( min_st, agv_st, max_st ) }
    

def get_odo( data ):
    """ -------------------------------------------------------------------------------------------------------------
    get the odometry a given sample from the nuScenes database

    data:           [dir] nuScenes dictionary of a sweep data

    return:         [tuple] <speed> <steer>

    ------------------------------------------------------------------------------------------------------------- """
    time        = data[ 'timestamp' ]                               # camera data token
    diff        = time                                              # largest possible time difference
    # search the can bus message with time stamp closest to the current sweep frame
    for i, m in enumerate( can_msg ):
        if abs( m[ 'utime' ] - time ) < diff:
            diff    = abs( m[ 'utime' ] - time )
            i_min   = i
    speed       = can_msg[ i_min ][ 'vel' ][ 0 ]               # forward speed
    steer       = can_msg[ i_min ][ 'rotation_rate' ][ -1 ]    # rotation rate

    return speed, steer
    

def get_frame_data( data ):
    """ -------------------------------------------------------------------------------------------------------------
    get the necessary data for a given sample from the nuScenes database
    see https://www.nuscenes.org/data-format for the various dictionary keys used here

    data:           [dir] nuScenes dictionary of a sweep data

    return:         [tuple] <cam_trans> <cam_rot> <cam_model> <ego_trans> <ego_rot>

    ------------------------------------------------------------------------------------------------------------- """
    cam_token   = data[ 'calibrated_sensor_token' ]                 # camera data token
    cam_data    = nu.get( 'calibrated_sensor', cam_token )          # data related with the camera in this frame
    cam_trans   = np.array( cam_data[ 'translation' ] )             # camera 3D translation
    cam_rot     = Quaternion( cam_data[ 'rotation' ] )              # camera 3D rotation
    cam_model   = np.array( cam_data[ 'camera_intrinsic' ] )        # camera intrinsic model
    ego_token   = data[ 'ego_pose_token' ]                          # ego pose token
    ego_pose    = nu.get( 'ego_pose', ego_token )                   # ego pose during this frame
    ego_trans   = np.array( ego_pose[ 'translation' ] )             # ego 3D translation
    ego_rot     = Quaternion( ego_pose[ 'rotation' ] )              # ego 3D rotation

    return cam_trans, cam_rot, cam_model, ego_trans, ego_rot
    

def track_frame( data_token, frame, dest ):
    """ -------------------------------------------------------------------------------------------------------------
    store all tracking data for a single frame
    for the output files naming conventions are the following, taking as example the frame #18:

    image with b/w heatmap of all 2D b.b.
        ht_018.jpg

    image with RGB content heatmap of all 2D b.b.
        hc_018.jpg

    image with linear occupancy grid of all 2D b.b.
        og_018.jpg

    image with log-occupancy grid of all 2D b.b.
        ol_018.jpg

    data_token:     [str] the identifier (token string) of a sweep data in the nuScenes database
    frame:          [int] progressive number of this frame in the processed sequence
    dest:           [str] path prefix for the folder where to write output files

    return:         [str] pathname of the original frame
    ------------------------------------------------------------------------------------------------------------- """
    data        = nu.get( 'sample_data', data_token )               # the dictionary with all data information
    img         = os.path.join( dir_orig, data[ 'filename' ] )      # pathname of the original frame
    shutil.copy( img, dest )                                        # make a local copy of the original frame

    sample      = nu.get( 'sample', data[ 'sample_token' ] )        # the sample (keyframe) closest to this sweep
    scene_id    = int( nu.get( 'scene', sample[ 'scene_token' ] )[ 'name' ].split( '-' )[ -1 ] )

    # get all what we need for this sample
    cam_trans, cam_rot, cam_model, ego_trans, ego_rot  = get_frame_data( data )
    cam         = cam_trans, cam_rot, cam_model
    ego         = ego_trans, ego_rot

    if DO_ATTN:
        img_curr    = Image.open( img )                             # image of the frame
        img_heat    = Image.new( 'L', img_curr.size, color=0 )      # heatmap image of the frame, with same size
        img_blck    = Image.new( 'RGB', img_curr.size, color=0 )    # black image used for heatmap content
        drw_heat    = ImageDraw.Draw( img_heat )                    # heatmap drawing area

    if DO_GRID:
        img_ogrd    = Image.new( 'L', grid_size, color=0 )          # linear occupancy grid image of the frame
        drw_ogrd    = ImageDraw.Draw( img_ogrd )                    # linear occupancy grid drawing area
        img_oglg    = Image.new( 'L', warp_size, color=0 )          # log-occupancy grid image of the frame
        drw_oglg    = ImageDraw.Draw( img_oglg )                    # log-occupancy grid drawing area

    if DO_ODOM:
        img_o   = str_frm_odm.format( scene_id, frame )             # filename for odometry
        img_o   = os.path.join( dest, img_o )                       # pathname for odometry
        sp, st  = get_odo( data )                                   # get speed and steering angle
        np.save( img_o, np.array( ( sp, st ) ) )                    # write odometry on file

    if data[ 'is_key_frame' ]:                                      # if this sweep is a keyframe
        boxes   = list( map( nu.get_box, sample[ 'anns' ] ) )       # get all objects boxes in the ordinary way
    else:                                                           # otherwise
        boxes   = nu.get_boxes( data_token )                        # use this interpolating function

    for bb in boxes:
        if bb.name not in nuscenes_obj:                             # check if is not a car-like object
            continue                                                # in case discard it
        bbs         = compute_boxes( cam, ego, bb )                 # get object bounding boxes
        if bbs is None:                                             # if bounding boxes get out of view
            continue                                                # discard them
        b2, b3      = bbs                                           # current bounding boxes

        if DO_ATTN:
            drw_heat.rectangle( b2, outline=None, fill='white')     # draw the 2D bounding box on heatmap
        if DO_GRID:
            fill_grid( drw_ogrd, b3, log_grid=False )               # draw in the linear occupancy grid
            points  = project_warp( b3 )                            # compute log transformation
            if points is not None:
                fill_grid( drw_oglg, points, log_grid=True )        # draw in the log-occupancy grid

    if DO_ATTN:
        img_o   = str_frm_loc.format( scene_id, frame )             # filename for the heatmap image
        img_o   = os.path.join( dest, img_o )                       # pathname for the heatmap image
        gauss   = ImageFilter.GaussianBlur( radius=4 )              # Gaussian blurring
        img_ht  = img_heat.filter( gauss )                          # smooth the heatmap image
        img_ht.save( img_o )                                        # save the heatmap image
        img_hc  = Image.composite( img_curr, img_blck, img_ht )     # image with RGB content in the heatzones
        img_o   = str_frm_att.format( scene_id, frame )             # filename for the heatmap content image
        img_o   = os.path.join( dest, img_o )                       # pathname for the heatmap content image
        img_hc.save( img_o )                                        # save the heatmap content image
    if DO_GRID:
        img_o   = str_frm_occ.format( scene_id, frame )             # filename for the linear occupancy image
        img_o   = os.path.join( dest, img_o )                       # pathname for the linear occupancy image
        img_ogrd.save( img_o )                                      # save the linear occupancy image
        if DO_MASK:
            mask_grid( drw_oglg, log_grid=True )                    # draw a mask in the log-occupancy grid
        img_o   = str_frm_wrp.format( scene_id, frame )             # filename for the log occupancy image
        img_o   = os.path.join( dest, img_o )                       # pathname for the log occupancy image
        img_oglg.save( img_o )                                      # save the log occupancy image

    return img


def track_frames( seq ):
    """ -------------------------------------------------------------------------------------------------------------
    process a nuScenes sequence, saving b.b. contents, and odometry
    loops on all "sweeps" frames

        seq:            [int] number of the nuScenes sequence
    ------------------------------------------------------------------------------------------------------------- """
    global can_msg                                             # CAN bus messages of the entire sequence

    sample_token    = nu.scene[ seq ][ 'first_sample_token' ]       # token of the first sample in the sequence
    scene_name      = nu.scene[ seq ][ 'name' ]                     # name of the sequence
    sample          = nu.get( 'sample', sample_token )
    data_token      = sample[ 'data' ][ nuscenes_cam ]              # token of the frame image
    dst_dir         = os.path.join( dir_new, scene_name )     # destination directory
    if REPLACE and os.path.exists( dst_dir ):
        shutil.rmtree( dst_dir )
    if not os.path.exists( dst_dir ):
        os.makedirs( dst_dir )

    can_msg    = nc.get_messages( scene_name, 'pose' )         # get odometry at 50Hz
    odo_stat        = stat_odo()                                    # odometry statistics
    str_frm_odmo         = "{:^7} {:7.4f} {:7.4f} {:7.4f}\n"             # odometry statistics formatting
    with open( os.path.join(dst_dir, fname_odom), 'w' ) as f:        # open the file for odometry statistics
        for q in odo_stat:
            f.write( str_frm_odmo.format( q, *odo_stat[ q ] ) )          # write the odometry statistics

    if DO_FLST:
        f           = open( os.path.join(dst_dir, fname_img), 'w' ) # open the file for writing frame filenames
    # loop over all frames in the sequence
    frame   = 0
    while len( data_token ):                                        # the sequence ends with an empty token
        frame       += 1                                            # enumerate the frames in the sequence
        img         = track_frame( data_token, frame, dst_dir )     # do the dirty job
        if DO_FLST:
            f.write( img + '\n' )
        data_token  = nu.get( 'sample_data', data_token )['next']   # advance to the next sample
    
    if DO_FLST:
        f.close()


def track_nuscenes():
    """ -------------------------------------------------------------------------------------------------------------
    retrieve all nuScenes sequences of interest

    ------------------------------------------------------------------------------------------------------------- """
    for seq in nuscenes_seqs:
        name        = nu.scene[ seq ][ 'name' ]
        scene_id    = int( name.split( '-' )[ -1 ] )
        if scene_id in nc.can_blacklist:                                 # skip sequences missing can bus data
            if verbose:
                print( f"sequence {seq} (with name {name}) is in can_blacklist, skipping..." )
            continue
        if verbose:
            print( "processing sequence {:04d}...".format( seq ) )
        track_frames( seq )
    

def summary():
    """ -------------------------------------------------------------------------------------------------------------
    write the description of all scenes

    ------------------------------------------------------------------------------------------------------------- """
    with open( os.path.join( dir_new, "scenes.txt" ), 'w' ) as f:
        for i, scene in enumerate( nu.scene ):
            f.write( "{:04d} {}\n".format( i, scene[ 'description' ] ) )
    

# ===================================================================================================================
#
#   MAIN
#
#   NOTE to be executed from the main folder (above 'src' and 'dataset')
#
# ===================================================================================================================
if __name__ == '__main__':
    if DO_EXEC:
        print( "Starting nuScenes extraction." )
        nu, nc  = init_nu()
        track_nuscenes()
