"""
#####################################################################################################################

    Delft visiting project
    Alice   2020

    Functions for parsing large datasets during training in Keras

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
import      tensorflow      as tf
from        keras           import backend  as K
from        keras.utils     import Sequence
os.environ[ 'PYTHONHASHSEED' ] = str( SEED )
np.random.seed( SEED )
rn.seed( SEED )
tf.set_random_seed( SEED )
# -------------------------------------------------------------------------------------------------------------------

import      datetime
from        PIL             import Image
from        print_msg       import print_err, print_wrn

DEBUG       = False

# classes of data in nuScenes, with condition for selection and dimension of the data
class_code          = {
        'ATT':  { 'cond' : lambda x: x.startswith( 'att_' ),    'size' : ( 900, 1600, 3 ) },    # attention map
        'LOC':  { 'cond' : lambda x: x.startswith( 'loc_' ),    'size' : ( 900, 1600, 1 ) },    # location map
        'FRM':  { 'cond' : lambda x: '__CAM_FRONT__' in x,      'size' : ( 900, 1600, 3 ) },    # full image frame
        'OCC':  { 'cond' : lambda x: x.startswith( 'occ_' ),    'size' : ( 128,  128, 1 ) },    # occupancy grid
        'WRP':  { 'cond' : lambda x: x.startswith( 'wrp_' ),    'size' : ( 128,  128, 1 ) },    # warped occupancy grid
        'ODM':  { 'cond' : lambda x: x.startswith( 'odm_' ),    'size' : (   2, ) }             # odometry values
}

frac_train          = 0.70                                      # fraction of training set
frac_valid          = 0.25                                      # fraction of validation set
frac_test           = 0.05                                      # fraction of test set
assert ( frac_train + frac_valid + frac_test - 1 ) < 0.001

file_exten          = ( '.jpg', '.png', '.npy' )                # accepted data file formats
cond_exten          = lambda x: x.endswith( file_exten )        # condition selecting only relevant files



# ===================================================================================================================
#
#   Generator of batches of data for training
#   It works by reading the path of samples stored in "data class" dictionaries
#   The data classes specify the kind of input/output, like for example attention maps, occupancy grids, etc.
#
# ===================================================================================================================

class DataGenerator( Sequence ):

    def __init__( self, sample_ids, input_dicts, target_dicts, input_sizes, target_sizes, batch_size, shuffle ):
        """ ---------------------------------------------------------------------------------------------------------
        Initialization

        sample_ids:     [list of str] IDs of samples to use for batch creation
        input_dicts:    [list of dict] input values of samples (key = sample ID, value = path to file) for each class
        target_dicts:   [list of dict] target values of samples (key = sample ID, value = path to file) for each class
        input_sizes:    [list of tuple] sizes of inputs for each class
        targe_size:     [list of tuple] sizes of targets for each class
        batch_size:     [int] batch size
        shuffle:        [bool] if True, at each epoch, the samples are randomly distributed in the batches
        --------------------------------------------------------------------------------------------------------- """
        for d in input_dicts + target_dicts:
            # all class dicts must share the same IDs
            assert input_dicts[ 0 ].keys() == d.keys()

        # all samples IDs must be contained in each class dicts
        assert set( sample_ids ).issubset( input_dicts[ 0 ].keys() )

        assert len( input_dicts ) == len( input_sizes )
        assert len( target_dicts ) == len( target_sizes )

        self.sample_ids     = sample_ids                    # computed by retrieve_samples()
        self.input_dicts    = input_dicts                   # computed by retrieve_samples()
        self.target_dicts   = target_dicts                  # computed by retrieve_samples()
        self.input_sizes    = input_sizes
        self.target_sizes   = target_sizes
        self.batch_size     = batch_size
        self.shuffle        = shuffle

        self.n_samples      = len( self.sample_ids )
        self.n_inputs       = len( self.input_dicts )
        self.n_targets      = len( self.target_dicts )

        self.on_epoch_end()                                 # initialize exploration indices



    def on_epoch_end( self ):
        """ ---------------------------------------------------------------------------------------------------------
        Updates sample indices after each epoch.
        If the shuffle parameter is set to True, we will get a new order of exploration at each pass (or just
        keep a linear exploration scheme otherwise).
        --------------------------------------------------------------------------------------------------------- """
        self.sample_idxs    = np.arange( self.n_samples )   # sample indices

        if self.shuffle == True:
            np.random.shuffle( self.sample_idxs )



    def __len__( self ):
        """ ---------------------------------------------------------------------------------------------------------
        Return the number of batches per epoch
        --------------------------------------------------------------------------------------------------------- """
        n_batches   = self.n_samples / self.batch_size
        return int( np.floor( n_batches ) )



    def __getitem__( self, batch_idx ):
        """ ---------------------------------------------------------------------------------------------------------
        Return the data of a batch

        batch_idx:      [int] batch index

        return:         [list] input and target values for each sample of the batch
        --------------------------------------------------------------------------------------------------------- """
        # generate sample indices of the batch
        i1                  = self.batch_size * batch_idx
        i2                  = self.batch_size * ( batch_idx + 1 )
        batch_sample_idxs   = self.sample_idxs[ i1 : i2 ]

        # find corresponding sample IDs
        batch_sample_ids    = [ self.sample_ids[ k ] for k in batch_sample_idxs ]

        # initialize arrays
        x   = [ np.empty( ( self.batch_size, *s ) ) for s in self.input_sizes ]
        y   = [ np.empty( ( self.batch_size, *s ) ) for s in self.target_sizes ]

        # load data
        for i, sample_id in enumerate( batch_sample_ids ):
            for c in range( self.n_inputs ):
                x[ c ][ i, ]    = self.__load_sample( sample_id, self.input_dicts[ c ] )

            for c in range( self.n_targets ):
                y[ c ][ i, ]    = self.__load_sample( sample_id, self.target_dicts[ c ] )

        if self.n_inputs == 1:  x = x[ 0 ]
        if self.n_targets == 1: y = y[ 0 ]

        return [ x, y ]



    def __load_sample( self, sample_id, class_dict ):
        """ ---------------------------------------------------------------------------------------------------------
        Given a sample ID and a dictionary of a single class, return the value of the sample for that class

        sample_id:      [str] sample ID
        class_dict:     [dict] key = sample ID, value = path to file

        return:         [np.array] sample value
        --------------------------------------------------------------------------------------------------------- """
        s   = class_dict[ sample_id ]                                   # path to file
        
        # image case
        if s.endswith( ( '.jpg', '.png' ) ):    

            if DEBUG:                                                   # save file as check
                shutil.copy( s, 'tmp' )

            a   = np.array( Image.open( s ) )
            a   = a.astype( float ) / 255                               # normalize between 0..1
            if len( a.shape ) == 2: a = np.expand_dims( a, axis=-1 )    # add channel in grayscale images
            return a

        # npy case
        if s.endswith( '.npy' ):
            a   = np.load( s )                                          # TODO CHANNEL CODING!
            return a

        print_err( "File extension not recognized: {}".format( s ) )



# ===================================================================================================================
#
#   - get_list_sequence
#   - retrieve_samples
#   - build_generator
#
# ===================================================================================================================

def get_list_sequence( src_dir, black_list=None, n_seq=None ):
    """ -------------------------------------------------------------------------------------------------------------
    Remove bad sequences from dataset list
    Consider number of black images, stationary videos and night/rain conditions

    src_dir:        [str] root of dataset (containing sub-folders for each sequence)
    black_list:     [str] filename to black list file (if None use all)
    n_seq:          [int] number of sequence sub-folders to use in the dataset (if None use all)

    return:         [list of str] selected sequences
    ------------------------------------------------------------------------------------------------------------- """
    list_seqs       = []

    if isinstance( black_list, str ):
        # this dict is produced by check_seq() in extract_nu.py
        d           = np.load( black_list, allow_pickle=True ).item()
        list_seqs   = [ k for k in sorted( d, key=d.get, reverse=True ) if d[ k ][ 1 ] == ( False, False, False ) ]
    else:
        list_seqs   = [ d for d in sorted( os.listdir( src_dir ) ) if os.path.isdir( os.path.join( src_dir, d ) ) ]

    if isinstance( n_seq, int ):
        list_seqs   = list_seqs[ :n_seq ]

    return list_seqs



def retrieve_samples( src_dir, class_names, class_conds, black_list=None, n_seq=None, shuffle=True ):
    """ -------------------------------------------------------------------------------------------------------------
    Create dictionaries with info to retrieve samples from the dataset.
    The dicts will be used by DataGenerator to create the data batches.

    The first dict organizes the sample IDs into dataset partitions (train/valid/test)
        parts_dict      = { 'train': [ 'id_1', 'id_2', ... ],
                            'valid': [ 'id_n', ... ],
                            'test':  [ 'id_m', ... ] }

    The other dicts (one for each data class passed as argument) link each sample ID with the path to the
    corresponding file of that class.
        class_dict_a    = { 'id_1': "path_to_attention_map_1", 'id_2': "path_to_attention_map_2", ... }
        class_dict_b    = { 'id_1': "path_to_occupancy_grid_1", 'id_2': "path_to_occupancy_grid_2", ... }
        class_dict_c    = { ... }

    src_dir:        [str] root of dataset (containing sub-folders for each sequence)
    class_names:    [list of str] names of the data classes
    class_conds:    [list of functions] conditions for selecting files for each sub-class
    black_list:     [str] filename to black list file (if None use all)
    n_seq:          [int] number of sequence sub-folders to use in the dataset (if None use all)
    shuffle:        [bool] if True, the dataset is partitioned in random order of samples

    return:         [dict] key = name of dataset partition, value = list of sample IDs
                    [list of dict] one for each class, key = sample ID, value = path to sample file
    ------------------------------------------------------------------------------------------------------------- """
    sample_id       = 'sample_{:06d}'                               # format of sample id
    n_class         = len( class_names )
    set_ids         = set()                                         # set of unique sample ids generated

    # one dict for each class (key = sample ID, value = path to file)
    class_dicts     = [ {} for i in range( n_class ) ]
    class_cnt       = n_class * [ 1 ]                               # a separate ID counter for each class

    # key = partition of dataset, value = list of sample IDs
    parts_dict      = { 'train' : None, 'valid' : None, 'test' : None }

    # list of all sequence sub-folders to use
    list_seqs       = get_list_sequence( src_dir, black_list=black_list, n_seq=n_seq )

    # fill the dicts parsing all the files (once) of the dataset
    for dr in list_seqs:
        for f in sorted( os.listdir( os.path.join( src_dir, dr ) ) ):
            for c in range( n_class ):
                if class_conds[ c ]( f ) and cond_exten( f ):
                    i                       = sample_id.format( class_cnt[ c ] )
                    class_dicts[ c ][ i ]   = os.path.join( src_dir, dr, f )
                    set_ids.add( i )
                    class_cnt[ c ]          += 1
                    continue                            # a file belongs to only one class

    for dt in class_dicts:
        assert set_ids == dt.keys()                     # all class dicts must share the same keys (sample ids)
    n_ids   = len( set_ids )

    # choose whether to shuffle the sample order
    list_ids                = sorted( list( set_ids ) )
    if shuffle:             rn.shuffle( list_ids )

    # partition list of sample IDs into the 3 sets for train/valid/test
    f_tr                    = int( frac_train * n_ids )
    f_vd                    = int( frac_valid * n_ids )
    f_ts                    = int( frac_test * n_ids )
    parts_dict[ 'train' ]   = list_ids[ 0           : f_tr ]
    parts_dict[ 'valid' ]   = list_ids[ f_tr        : f_tr + f_vd ]
    parts_dict[ 'test' ]    = list_ids[ f_tr + f_vd : f_tr + f_vd + f_ts ]

    return parts_dict, class_dicts



def build_generator( dataset_dir, input_class, target_class, batch_size, black_list=None, n_seq=None,
        shuffle_batch=True, shuffle_part=True ):
    """ -------------------------------------------------------------------------------------------------------------
    Build the generators producing batches of data for training and validation.
    The types (classes) of input and target data are passed by arguments.

    dataset_dir:    [str] root of dataset (containing sub-folders for each sequence)
    input_class:    [str or list of str] class code(s) of input data
    target_class:   [str or list of str] class code(s) of target data
    batch_size:     [int] batch size
    black_list:     [str] filename to black list file (if None use all)
    n_seq:          [int] number of sequence sub-folders to use in the dataset (if None use all)
    shuffle_batch:  [bool] if True, at each epoch, the samples are randomly distributed in the batches
    shuffle_part:   [bool] if True, the dataset is partitioned randomly in train/valid/test

    return:         [DataGenerator] 3 generators
    ------------------------------------------------------------------------------------------------------------- """
    if isinstance( input_class, str ):          # convert to list, so the rest of the code is the same in both cases
        input_class = [ input_class, ]

    if isinstance( target_class, str ):         # convert to list, so the rest of the code is the same in both cases
        target_class = [ target_class, ]

    n_inp   = len( input_class )
    n_trg   = len( target_class )
    cls     = [ *input_class, *target_class ]
    cnd     = [ class_code[ c ][ 'cond' ] for c in cls ]

    assert set( cls ).issubset( class_code.keys() )

    parts_dict, class_dicts     = retrieve_samples( dataset_dir, cls, cnd, black_list=black_list, n_seq=n_seq,
            shuffle=shuffle_part )

    train_ids                   = parts_dict[ 'train' ]     # training sample IDs
    valid_ids                   = parts_dict[ 'valid' ]     # validation sample IDs
    test_ids                    = parts_dict[ 'test' ]      # test sample IDs
    input_d                     = class_dicts[ :n_inp ]     # dicts with paths to files of input classes
    target_d                    = class_dicts[ n_inp: ]     # dicts with paths to files of target classes
    input_s                     = [ class_code[ c ][ 'size' ] for c in input_class ]
    target_s                    = [ class_code[ c ][ 'size' ] for c in target_class ]

    train_gen   = DataGenerator( train_ids, input_d, target_d, input_s, target_s, batch_size, shuffle=shuffle_batch )
    valid_gen   = DataGenerator( valid_ids, input_d, target_d, input_s, target_s, batch_size, shuffle=shuffle_batch )
    test_gen    = DataGenerator( test_ids, input_d, target_d, input_s, target_s, batch_size, shuffle=shuffle_batch )

    return train_gen, valid_gen, test_gen
