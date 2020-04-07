"""
#####################################################################################################################

    Delft visiting project
    Alice   2020

    Training of a model

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

import  sys
import  datetime
from    math                import ceil
import  matplotlib
matplotlib.use( 'agg' )     # to use matplotlib with unknown 'DISPLAY' var (when using remote display)
from    matplotlib          import pyplot   as plt



# ===================================================================================================================
#
#   - set_callback
#   - train_model
#   - plot_history
#
# ===================================================================================================================

def set_callback( filename, period=25 ):
    """ -------------------------------------------------------------------------------------------------------------
    Save the model after every epoch

    filename:           [str] where to save the model
    period:             [int] interval (number of epochs) between checkpoints

    return:             [keras.callbacks.ModelCheckpoint]
    ------------------------------------------------------------------------------------------------------------- """
    calls   = []

    # NOTE BE CAREFUL! Saving checkpoints is a CPU operation. If you do it too often, you cause a CPU bottleneck,
    # and then the training gets super slow because the GPUs almost never work (case when Volatile GPU-Util = 0%)

    calls.append( callbacks.ModelCheckpoint(
            filename,
            save_best_only          = True,
            save_weights_only       = True,
            period                  = period
    ) )

    return calls



def train_model( model, model_file, generators, n_epochs, batch_size, n_gpus, process_queue=10, process_multi=False,
        process_workers=1, shuffle=True, chckpnt=0 ):
    """ -------------------------------------------------------------------------------------------------------------
    Training procedure

    model:                  [keras.models.Model] compiled model
    model_file:             [str] where to save the model
    generators:             [list of DataGenerator] train and valid generators
    n_epochs:               [int] number of epochs
    batch_size:             [int] batch size
    n_gpus                  [int] number of GPUs (0 if CPU)
    process_queue:          [int] max size for the generator queue
    process_multi:          [bool] use process-based threading
    process_workers:        [int] max number of processes
    shuffle:                [bool] shuffle the order of the batches at the beginning of each epoch
    chckpnt:                [int] interval (number of epochs) between saving checkpoints

    return:                 [keras.callbacks.History], [datetime.timedelta]
    ------------------------------------------------------------------------------------------------------------- """
    train_gen, valid_gen            = generators
    n_train, n_valid                = train_gen.n_samples, valid_gen.n_samples
    train_steps                     = ceil( n_train / batch_size )
    valid_steps                     = ceil( n_valid / batch_size )

    # train using multiple GPUs
    if n_gpus > 1:
        model   = utils.multi_gpu_model( model, gpus=n_gpus )

    callbacks   = set_callback( model_file, period=chckpnt ) if chckpnt > 0 else None

    t_start     = datetime.datetime.now()                               # starting time of execution
    hist        = model.fit_generator(
            train_gen,
            epochs                  = n_epochs,
            validation_data         = valid_gen,
            steps_per_epoch         = train_steps,
            validation_steps        = valid_steps,
            callbacks               = callbacks,
            max_queue_size          = process_queue,                    # max size for the generator queue (default=10)
            use_multiprocessing     = process_multi,                    # use process-based threading (default=False)
            workers                 = process_workers,                  # max number of processes (default=1)
            shuffle                 = shuffle,
            verbose                 = 2
    )
    t_end       = datetime.datetime.now()                               # ending time of execution

    # best results in history
    indx        = np.argmin( hist.history[ 'val_loss' ] )
    best_loss   = hist.history[ 'loss' ][ indx ]
    best_val    = hist.history[ 'val_loss' ][ indx ]
    print( "Best model reached:\tloss: {:.3f}\tval_loss: {:.3f}\n".format( best_loss, best_val ) )

    return hist, ( t_end - t_start )



def plot_history( history, fname ):
    """ -------------------------------------------------------------------------------------------------------------
    Plot the loss performance, on training and validation sets

    history:        [keras.callbacks.History]
    fname:          [str] path+name of output file without extension
    ------------------------------------------------------------------------------------------------------------- """
    train_loss  = history.history[ 'loss' ]
    valid_loss  = history.history[ 'val_loss' ]
    epochs      = range( 1, len( train_loss ) + 1 )

    plt.plot( epochs, train_loss, 'r--' )
    plt.plot( epochs, valid_loss, 'b-' )

    plt.legend( [ 'Training Loss', 'Validation Loss' ] )
    plt.xlabel( 'Epoch' )
    plt.ylabel( 'Loss' )

    plt.grid( True )
    plt.rc( 'grid', linestyle='--', color='lightgrey' )
    plt.savefig( "{}.pdf".format( fname ) )

    if len( train_loss ) > 5:
        m   = np.mean( train_loss )
        s   = np.std( train_loss )
        plt.ylim( [ m - s, m + s ] )
        plt.grid( True )
        plt.savefig( "{}_zoom.pdf".format( fname ) )

    plt.close()
