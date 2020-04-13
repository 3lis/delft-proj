"""
#####################################################################################################################

    Delft visiting project
    Alice   2020

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
os.environ[ 'PYTHONHASHSEED' ] = str( SEED )
np.random.seed( SEED )
rn.seed( SEED )
tf.set_random_seed( SEED )
# -------------------------------------------------------------------------------------------------------------------

import      sys
import      time
import      datetime
import      pickle

import      train_net       as tr
import      test_net        as ts
import      arch_net        as ar
import      load_cnfg       as lc
import      load_model      as lm
import      data_gen        as dg
from        print_msg       import print_err, print_wrn, print_flush


FRMT                = "%y-%m-%d_%H-%M-%S"                           # datetime format for folder names

dataset_dir         = "dataset/nuScenes/data"                       # dataset of processed data
dataset_black_list  = "dataset/nuScenes/black_list.npy"             # dict used to exclude bad sequence from dataset

# folders and files inside the main execution folder - NOTE the variables will be updated in init_dirs()
dir_current         = None
dir_res             = 'res'
dir_log             = 'log'
dir_src             = 'src'
dir_cnfg            = 'cnfg'
dir_plot            = 'plot'
dir_test            = 'test'
log_train           = "train.log"
log_hist            = "hist.pickle"
log_cnfg            = "cnfg.pickle"
log_err             = "err.log"
log_time            = "time.log"
nn_best             = "nn_best.h5"
nn_final            = "nn_final.h5"

cnfg                                        = None                  # [Config] object keeping together all parameters
n_gpus                                      = None                  # [int] number of visible GPUs
train_gen, valid_gen, test_gen              = None, None, None      # [DataGenerator]
nn                                          = None                  # network object


def init_cnfg():
    """ -------------------------------------------------------------------------------------------------------------
    Set global parameters from command line and python file
    ------------------------------------------------------------------------------------------------------------- """
    global cnfg

    cnfg            = lc.Config()

    # load parameters from command line
    line_kwargs     = lc.read_args()
    cnfg.load_from_line( **line_kwargs )

    # load parameters from file
    # NOTE it's a mess to import some module from a parent directory, so 'cnfg/' must stay inside 'src/'
    mdl             = "cnfg." + cnfg.CONFIG                         # name of python module
    exec( "import " + mdl )                                         # exec the import statement
    file_kwargs     = eval( mdl + ".kwargs" )                       # assign the content to a variable
    cnfg.load_from_file( **file_kwargs )

    # save the Config object
    with open( log_cnfg, 'wb') as f:
        pickle.dump( cnfg, f )



def init_gpu():
    """ -------------------------------------------------------------------------------------------------------------
    Set GPU-related parameters
    ------------------------------------------------------------------------------------------------------------- """
    global n_gpus

    n_gpus          = eval( cnfg.GPU )

    # CUDA_VISIBLE_DEVICES accepts string like "1, 4, 5"
    if isinstance( n_gpus, int ):
        os.environ[ "CUDA_VISIBLE_DEVICES" ]    = str( list( range( n_gpus ) ) )[ 1 : -1 ]
    elif isinstance( n_gpus, ( tuple, list ) ):
        os.environ[ "CUDA_VISIBLE_DEVICES" ]    = str( n_gpus )[ 1 : -1 ]
        n_gpus                                  = len( n_gpus )
    else:
       print_err( "GPUs specification {} not valid".format( n_gpus ) )

    # GPU memory fraction
    if n_gpus > 0:
        tf_cnfg                                             = tf.ConfigProto()
        tf_cnfg.gpu_options.per_process_gpu_memory_fraction = cnfg.FGPU
        tf_session                                          = tf.Session( config=tf_cnfg )
        K.set_session( tf_session )



def init_dirs():
    """ -------------------------------------------------------------------------------------------------------------
    Set paths to directories where to save the execution
    ------------------------------------------------------------------------------------------------------------- """
    global dir_current, dir_log, dir_plot, dir_src, dir_cnfg, dir_test                      # dirs
    global log_train, log_time, log_err, log_hist, log_cnfg, nn_best, nn_final              # files

    dir_current     = os.path.join( dir_res, time.strftime( FRMT ) )
    dir_log         = os.path.join( dir_current, dir_log )
    dir_src         = os.path.join( dir_current, dir_src )
    dir_cnfg        = os.path.join( dir_src, dir_cnfg )
    dir_plot        = os.path.join( dir_current, dir_plot )
    dir_test        = os.path.join( dir_current, dir_test )

    os.makedirs( dir_current )
    os.makedirs( dir_log )
    os.makedirs( dir_src )
    os.makedirs( dir_cnfg )
    os.makedirs( dir_plot )
    os.makedirs( dir_test )

    log_train       = os.path.join( dir_log, log_train )
    log_hist        = os.path.join( dir_log, log_hist )
    log_cnfg        = os.path.join( dir_log, log_cnfg )
    log_err         = os.path.join( dir_log, log_err )
    log_time        = os.path.join( dir_log, log_time )
    nn_best         = os.path.join( dir_current, nn_best )
    nn_final        = os.path.join( dir_current, nn_final )



def archive():
    """ -------------------------------------------------------------------------------------------------------------
    Archive python source code and configuration file
    ------------------------------------------------------------------------------------------------------------- """
    pfile   = "src/*.py"
    os.system( "cp {} {}".format( pfile, dir_src ) )                        # save python sources
    os.system( "cp src/cnfg/{}.py {}".format( cnfg.CONFIG, dir_cnfg ) )     # save config file



def create_model():
    """ -------------------------------------------------------------------------------------------------------------
    Create the model object
    ------------------------------------------------------------------------------------------------------------- """
    global nn

    ar.TRAIN    = cnfg.TRAIN

    nn          = ar.create_model(
            cnfg.arch_code,
            cnfg.arch_kwargs,
            cnfg.enc_kwargs,
            cnfg.dec_kwargs,
            n_gpus,
            batch   = cnfg.batch_size,
            summary = dir_plot
    )

    s   = "Failed to load weights from {}."
    s   += "NOTE: You might want to check the correctness of 'valid_prfx' and 'invalid_prfx' in load_model.py"

    if cnfg.LOAD is not None:
        if not lm.load_h5( nn.model, cnfg.LOAD ):
            print_err( s.format( cnfg.LOAD ) )

    sys.stdout.flush()          # flush to have the current stdout in the log



def create_generators():
    """ -------------------------------------------------------------------------------------------------------------
    Create the dataset generators
    ------------------------------------------------------------------------------------------------------------- """
    global train_gen, valid_gen, test_gen

    bl      = dataset_black_list if cnfg.black_list else None

    generators                              = dg.build_generator(
            dataset_dir,
            cnfg.input_class,
            cnfg.target_class,
            cnfg.batch_size,
            black_list          = bl,
            n_seq               = cnfg.dataset_seq,
            shuffle_part        = cnfg.shuffle_partitions,
            shuffle_batch       = cnfg.shuffle_batch_samples
    )

    train_gen, valid_gen, test_gen          = generators

    frm1        = "{:<30}{:10d}\t{:10d}"
    frm2        = "{:<30}{:10d}"
    print( '\n' )
    print( frm1.format( "Train samples and batches:", train_gen.n_samples, train_gen.__len__() ) )
    print( frm1.format( "Valid samples and batches:", valid_gen.n_samples, valid_gen.__len__() ) )
    print( frm1.format( "Test samples and batches:", test_gen.n_samples, test_gen.__len__() ) )
    print( frm2.format( "Total samples:", train_gen.n_samples + valid_gen.n_samples + test_gen.n_samples ) )
    print_flush( 65 * '_' + '\n' )



def train_model():
    """ -------------------------------------------------------------------------------------------------------------
    Train the model
    ------------------------------------------------------------------------------------------------------------- """
    print_flush( "Now starting training...\n" )

    hist, train_time    = tr.train_model(
            nn.multi_model if n_gpus > 1 else nn.model,
            nn_best,
            ( train_gen, valid_gen ),
            cnfg.n_epochs,
            cnfg.batch_size,
            process_queue       = cnfg.process_queue,
            process_multi       = cnfg.process_multi,
            process_workers     = cnfg.process_workers,
            chckpnt             = cnfg.chckpnt,
            shuffle             = cnfg.shuffle_batch_order
    )

    # save model
    nn.model.save_weights( nn_final )       # NOTE even in case of n_gpus>1, nn.model is the one to be saved

    # save and plot history
    with open( log_hist, 'wb') as f:
        pickle.dump( hist.history, f )
    tr.plot_history( hist, os.path.join( dir_plot, 'loss' ) )

    # save duration of training
    with open( log_time, 'a' ) as f:
        f.write( "Training duration:\t{}\n".format( str( train_time ) ) )

    sys.stdout.flush()          # flush to have the current stdout in the log



def test_model():
    """ -------------------------------------------------------------------------------------------------------------
    Test the model
    ------------------------------------------------------------------------------------------------------------- """
    t_start     = datetime.datetime.now()

    ar.TRAIN    = False                             # TODO TODO TODO check
    ts.WARPED   = cnfg.target_class == 'WRP'        # TODO implement case of multi output class

    print_flush( "\n>>> TEST: Predictions on a batch of data\n" )
    ts.pred_batch( nn.model, test_gen, batch_indx=0, save=dir_test )

    print_flush( ">>> TEST: Evaluation stats\n" )
    ts.eval_dataset( nn.model, test_gen, thresh_iou=0.5, thresh_err=0.7, save=os.path.join( dir_test, "eval.txt" ) )

    # save duration of tests
    t_end       = datetime.datetime.now()
    with open( log_time, 'a' ) as f:
        f.write( "Testing duration:\t\t{}\n".format( str( t_end - t_start ) ) )

    

# ===================================================================================================================
#
#   MAIN
#
#   NOTE to be executed from the main folder (above 'src' and 'dataset')
#
# ===================================================================================================================
if __name__ == '__main__':

    t_start         = datetime.datetime.now()
    init_dirs()
    init_cnfg()
    init_gpu()

    # NOTE to restore use sys.stdout = sys.__stdout__
    if cnfg.REDIRECT:
        sys.stdout      = open( log_train, 'w' )
        sys.stderr      = open( log_err, 'w' )

    if cnfg.ARCHIVE:
        archive()

    create_model()
    create_generators()

    if cnfg.TRAIN:
        train_model()

    if cnfg.TEST:
        test_model()

    # save total duration
    t_end       = datetime.datetime.now()
    with open( log_time, 'a' ) as f:
        f.write( "Total duration:\t\t{}\n".format( str( t_end - t_start ) ) )

    print_flush( '~ End of execution! ~\n' )
