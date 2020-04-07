"""
#####################################################################################################################

    Delft visiting project
    Alice   2020

    Definition of the neural network architectures

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
from        keras           import models, layers, utils, optimizers, losses
from        keras           import backend      as K
os.environ[ 'PYTHONHASHSEED' ] = str( SEED )
np.random.seed( SEED )
rn.seed( SEED )
tf.set_random_seed( SEED )
# -------------------------------------------------------------------------------------------------------------------

from        print_msg       import print_err, print_wrn


TRAIN   = True                      # use to switch between the architecture for training or for inference
                                    # (in the variational case) -- it is set in main_exec.py

# Other globals and functions are placed after the definition of the classes

#####################################################################################################################
#
#   Classes
#
#   - Encoder
#   - Decoder
#   - EncoderDecoder
#
#####################################################################################################################

# ===================================================================================================================
#
#   Class for the generation of an decoder network
#
# ===================================================================================================================

class Encoder( object ):

    def __init__( self, summary=False, **kwargs ):
        """ ---------------------------------------------------------------------------------------------------------
        Initialization

        summary:                [str] path where to save plot if you want a summary+plot of model, False otherwise
        
        Expected parameters in kwargs:

        arch_layout:            [str] code describing the order of layers in the model
        input_size:             [list] height, width, channels
        net_indx:               [int] a unique index identifying the network

        conv_kernel_num:        [list of int] number of kernels for each convolution
        conv_kernel_size:       [list of int] (square) size of kernels for each convolution
        conv_strides:           [list of int] stride for each convolution
        conv_padding:           [list of str] padding (same/valid) for each convolution
        conv_activation:        [list of str] activation function for each convolution
        conv_train:             [list of bool] False to lock training of each convolution

        pool_size:              [list of int] pooling size for each MaxPooling
        
        enc_dnse_size:          [list of int] size of each dense layer
        enc_dnse_dropout:       [list of int] dropout of each dense layer
        enc_dnse_activation:    [list of str] activation function for each dense layer
        enc_dnse_train:         [list of bool] False to lock training of each dense layer
        --------------------------------------------------------------------------------------------------------- """
        for key, value in kwargs.items():
            setattr( self, key, value )

        # total num of convolutions, pooling and dense layers
        self.n_conv             = self.arch_layout.count( layer_code[ 'CONV' ] )
        self.n_pool             = self.arch_layout.count( layer_code[ 'POOL' ] )
        self.n_dnse             = self.arch_layout.count( layer_code[ 'DNSE' ] )

        # the last 3D shape before flattening, filled by _define_layers()
        self.last_3d_shape      = None

        # the keras.layers.Input object, filled by _define_encoder()
        self.input_layer        = None

        # the layout description must contain meaningful codes
        assert set( self.arch_layout ).issubset( layer_code.values() )

        # only a single flatten layer is accepted in the architecture
        assert self.arch_layout.count( layer_code[ 'FLAT' ] ) == 1

        assert self.n_dnse == len( self.enc_dnse_size ) == len( self.enc_dnse_dropout ) == \
                len( self.enc_dnse_activation ) == len( self.enc_dnse_train )
        assert self.n_conv == len( self.conv_kernel_num ) == len( self.conv_kernel_size ) == len( self.conv_strides ) == \
                len( self.conv_padding ) == len( self.conv_activation ) == len( self.conv_train )

        # create the network
        self.model_name     = 'encoder_{}'.format( self.net_indx )
        self.model          = self._define_encoder()

        if summary:
            model_summary( self.model, fname=os.path.join( summary, self.model_name ) )



    def _define_encoder( self ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the encoder model

        summary:        [bool] if True produce a summary of the model

        return:         [keras.models.Model] decoder model
        --------------------------------------------------------------------------------------------------------- """
        self.input_layer    = layers.Input( shape=self.input_size )
        model               = models.Model( 
                    inputs      = self.input_layer,
                    outputs     = self._define_layers( self.input_layer ),
                    name        = self.model_name
        )

        # TODO multi gpu model
        return model



    def _define_layers( self, x ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the network of layers

        x:              [tf.Tensor] input of the layers

        return:         [tf.Tensor] output of the layers
        --------------------------------------------------------------------------------------------------------- """
        i_conv, i_pool, i_dnse  = 3 * [ 0 ]                             # to keep count

        for i, layer in enumerate( self.arch_layout ):

            # convolutional layer
            if layer == layer_code[ 'CONV' ]:
                x       = self._conv2D( x, i_conv )
                i_conv  += 1

            # pooling layer
            elif layer == layer_code[ 'POOL' ]:
                x       = self._maxpool2D( x, i_pool )
                i_pool  += 1

            # dense layer
            elif layer == layer_code[ 'DNSE' ]:
                x       = self._dense( x, i_dnse )
                i_dnse  += 1

            # flat layer
            elif layer == layer_code[ 'FLAT' ]:
                self.last_3d_shape      = K.int_shape( x )[ 1: ]        # save the last 3D shape before flattening
                x       = layers.Flatten( name='flat_{}'.format( self.net_indx ) )( x )

            else:
                print_err( "Layer code '{}' not valid".format( layer ) )
                
        return x



    def _conv2D( self, x, indx ):
        """ ---------------------------------------------------------------------------------------------------------
        Create a Conv2D layer, using the parameters associated to the index passed as argument

            NOTE be aware of the difference between kernel_regularizer and activity_regularizer
            NOTE be aware there are biases also in convolutions

        x:              [tf.Tensor] input of layer
        indx:           [int] local index of layer

        return:         [tf.Tensor] layer output
        --------------------------------------------------------------------------------------------------------- """
        return layers.Conv2D(
                    self.conv_kernel_num[ indx ],                           # number of filters
                    kernel_size         = self.conv_kernel_size[ indx ],    # size of window
                    strides             = self.conv_strides[ indx ],        # stride (window shift)
                    padding             = self.conv_padding[ indx ],        # zero-padding around the image
                    activation          = self.conv_activation[ indx ],     # activation function
                    #kernel_initializer = self.conv_initializer,            # kernel initializer
                    #kernel_regularizer = self.conv_regularizer,            # kernel regularizer
                    #use_bias           = True,                             # convolutional biases
                    trainable           = self.conv_train[ indx ],
                    name                = 'conv_{}_{}'.format( self.net_indx, indx )
        )( x )



    def _maxpool2D( self, x, indx ):
        """ ---------------------------------------------------------------------------------------------------------
        Create a MaxPooling2D layer, using the parameters associated to the index passed as argument

        x:              [tf.Tensor] input of layer
        indx:           [int] local index of layer

        return:         [tf.Tensor] layer output
        --------------------------------------------------------------------------------------------------------- """
        return layers.MaxPooling2D(                          
                    pool_size       = self.pool_size[ indx ],               # pooling size
                    padding         = self.conv_padding[ indx ],            # zero-padding around the image
                    name            = 'pool_{}_{}'.format( self.net_indx, indx )
        )( x )



    def _dense( self, x, indx ):
        """ ---------------------------------------------------------------------------------------------------------
        Create a Dense layer, using the parameters associated to the index passed as argument

        x:              [tf.Tensor] input of layer
        indx:           [int] local index of layer

        return:         [tf.Tensor] layer output
        --------------------------------------------------------------------------------------------------------- """
        if self.enc_dnse_dropout[ indx ] > 0:                               # dropout
            x   = layers.Dropout( self.enc_dnse_dropout[ indx ] )( x )

        return layers.Dense(                          
                    self.enc_dnse_size[ indx ],                             # dimensionality of the output
                    activation      = self.enc_dnse_activation[ indx ],     # activation function
                    trainable       = self.enc_dnse_train[ indx ],
                    name            = 'dnse_{}_{}'.format( self.net_indx, indx )
        )( x )



# ===================================================================================================================
#
#   Class for the generation of an decoder network
#
# ===================================================================================================================

class Decoder( object ):

    def __init__( self, summary=False, **kwargs ):
        """ ---------------------------------------------------------------------------------------------------------
        Initialization

        summary:                [str] path where to save plot if you want a summary+plot of model, False otherwise

        Expected parameters in kwargs:

        arch_layout:            [str] code describing the order of layers in the model
        input_size:             [list] height, width, channels
        net_indx:               [int] a unique index identifying the network
        first_3d_shape:         [list] the first 3D shape after reshaping (height, width, channels)

        dcnv_kernel_num:        [list of int] number of kernels for each deconvolution
        dcnv_kernel_size:       [list of int] (square) size of kernels for each deconvolution
        dcnv_strides:           [list of int] stride for each deconvolution
        dcnv_padding:           [list of str] padding (same/valid) for each deconvolution
        dcnv_activation:        [list of str] activation function for each deconvolution
        dcnv_train:             [list of bool] False to lock training of each deconvolution

        dec_dnse_size:          [list of int] size of each dense layer
        dec_dnse_dropout:       [list of int] dropout of each dense layer
        dec_dnse_activation:    [list of str] activation function for each dense layer
        dec_dnse_train:         [list of bool] False to lock training of each dense layer
        --------------------------------------------------------------------------------------------------------- """
        for key, value in kwargs.items():
            setattr( self, key, value )

        # total num of deconvolutions and dense layers
        self.n_dcnv     = self.arch_layout.count( layer_code[ 'DCNV' ] )
        self.n_dnse     = self.arch_layout.count( layer_code[ 'DNSE' ] )

        # the layout description must contain meaningful codes
        assert set( self.arch_layout ).issubset( layer_code.values() )

        # only a single reshape layer is accepted in the architecture
        assert self.arch_layout.count( layer_code[ 'RSHP' ] ) == 1

        assert self.n_dnse == len( self.dec_dnse_size ) == len( self.dec_dnse_activation ) == \
                len( self.dec_dnse_dropout ) == len( self.dec_dnse_train )
        assert self.n_dcnv == len( self.dcnv_kernel_num ) == len( self.dcnv_kernel_size ) == len( self.dcnv_strides ) == \
                len( self.dcnv_padding ) == len( self.dcnv_activation ) == len( self.dcnv_train )

        # create the network
        self.model_name     = 'decoder_{}'.format( self.net_indx )
        self.model          = self._define_decoder()
        if summary:
            model_summary( self.model, fname=os.path.join( summary, self.model_name ) )



    def _define_decoder( self ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the decoder model

        summary:        [bool] if True produce a summary of the model

        return:         [keras.models.Model] decoder model
        --------------------------------------------------------------------------------------------------------- """
        x       = layers.Input( shape=self.input_size )
        model   = models.Model( 
                    inputs      = x,
                    outputs     = self._define_layers( x ),
                    name        = self.model_name
        )

        # TODO multi gpu model
        return model



    def _define_layers( self, x ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the network of layers

        x:              [tf.Tensor] input of the layers

        return:         [tf.Tensor] output of the layers
        --------------------------------------------------------------------------------------------------------- """
        i_dcnv, i_dnse          = 2 * [ 0 ]                             # to keep count

        for i, layer in enumerate( self.arch_layout ):

            # deconvolutional layer
            if layer == layer_code[ 'DCNV' ]:
                x       = self._deconv2D( x, i_dcnv )
                i_dcnv  += 1

            # dense layer
            elif layer == layer_code[ 'DNSE' ]:
                x       = self._dense( x, i_dnse )
                i_dnse  += 1

            # reshape layer
            elif layer == layer_code[ 'RSHP' ]:
                ts      = self.first_3d_shape
                x       = layers.Reshape( target_shape=ts, name='rshp_{}'.format( self.net_indx ) )( x )

            else:
                print_err( "Layer code '{}' not valid".format( layer ) )
                
        return x



    def _deconv2D( self, x, indx ):
        """ ---------------------------------------------------------------------------------------------------------
        Create a Conv2DTranspose layer, using the parameters associated to the index passed as argument

            NOTE be aware of the difference between kernel_regularizer and activity_regularizer
            NOTE be aware there are biases also in deconvolutions

        x:              [tf.Tensor] input of layer
        indx:           [int] local index of layer

        return:         [tf.Tensor] layer output
        --------------------------------------------------------------------------------------------------------- """
        return layers.Conv2DTranspose(
                    self.dcnv_kernel_num[ indx ],                           # number of filters
                    kernel_size         = self.dcnv_kernel_size[ indx ],    # size of window
                    strides             = self.dcnv_strides[ indx ],        # stride (window shift)
                    padding             = self.dcnv_padding[ indx ],        # zero-padding around the image
                    activation          = self.dcnv_activation[ indx ],     # activation function
                    #kernel_initializer = self.dcnv_initializer,            # kernel initializer
                    #kernel_regularizer = self.dcnv_regularizer,            # kernel regularizer
                    #use_bias           = True,                             # deconvolutional biases
                    trainable           = self.dcnv_train[ indx ],
                    name                = 'dcnv_{}_{}'.format( self.net_indx, indx )
        )( x )



    def _dense( self, x, indx ):
        """ ---------------------------------------------------------------------------------------------------------
        Create a Dense layer, using the parameters associated to the index passed as argument

        x:              [tf.Tensor] input of layer
        indx:           [int] local index of layer

        return:         [tf.Tensor] layer output
        --------------------------------------------------------------------------------------------------------- """
        if self.dec_dnse_dropout[ indx ] > 0:                                   # dropout
            x   = layers.Dropout( self.dec_dnse_dropout[ indx ] )( x )

        return layers.Dense(                          
                    self.dec_dnse_size[ indx ],                                 # dimensionality of the output
                    activation      = self.dec_dnse_activation[ indx ],         # activation function
                    trainable       = self.dec_dnse_train[ indx ],
                    name            = 'dnse_{}_{}'.format( self.net_indx, indx )
        )( x )

        

# ===================================================================================================================
#
#   Class for the generation of an encoding-decoding network
#
# ===================================================================================================================

class EncoderDecoder( object ):

    def __init__( self, enc_kwargs, dec_kwargs, summary=False, **arch_kwargs ):
        """ ---------------------------------------------------------------------------------------------------------
        Initialization

        summary:                [str] path where to save plot if you want a summary+plot of model, False otherwise

        Expected parameters in arch_kwargs:

        arch_layout:            [str] code describing the order of layers in the model
        input_size:             [list] height, width, channels
        optimiz:                [str] code of the optimizer
        loss_func:              [str] code of the loss function
        lrate:                  [float] learning rate
        --------------------------------------------------------------------------------------------------------- """
        for key, value in arch_kwargs.items():
            setattr( self, key, value )

        assert layer_code[ 'STOP' ] in self.arch_layout

        self.model_name     = "EncDec"
        self.model          = self._define_model( enc_kwargs, dec_kwargs, summary=summary )
        if summary:
            model_summary( self.model, fname=os.path.join( summary, self.model_name ) )



    def _define_model( self, enc_kwargs, dec_kwargs, summary=False ):
        """ ---------------------------------------------------------------------------------------------------------
        Create the encoder-decoder model

        enc_kwargs:     [dict] parameters for Encoder
        dec_kwargs:     [dict] parameters for Decoder
        summary:        [str] path where to save plot if you want a summary+plot of model, False otherwise

        return:         [keras.models.Model] model
        --------------------------------------------------------------------------------------------------------- """
        enc_kwargs[ 'net_indx' ]        = 1
        dec_kwargs[ 'net_indx' ]        = 2
        enc_kwargs[ 'arch_layout' ]     = self.arch_layout.split( layer_code[ 'STOP' ] )[ 0 ]
        dec_kwargs[ 'arch_layout' ]     = self.arch_layout.split( layer_code[ 'STOP' ] )[ -1 ]
        enc_kwargs[ 'input_size' ]      = self.input_size
        dec_kwargs[ 'input_size' ]      = ( enc_kwargs[ 'enc_dnse_size' ][ -1 ], )

        # create Encoder and Decoder objects
        enc                             = Encoder( summary=summary, **enc_kwargs )
        dec                             = Decoder( summary=summary, **dec_kwargs )

        # set network layers
        x       = enc.model( enc.input_layer )
        x       = dec.model( x )

        model   = models.Model( 
                    inputs      = enc.input_layer,
                    outputs     = x,
                    name        = self.model_name
        )

        # TODO multi gpu model
        return model



#####################################################################################################################
#
#   GLOBALs and other FUNCTIONs
#
#   - model_summary
#   - create_model
#
#####################################################################################################################

layer_code      = {                                     # one of the accepted type of layers
        'CONV':         'C',
        'DCNV':         'T',
        'DNSE':         'D',
        'FLAT':         'F',
        'POOL':         'P',
        'RSHP':         'R',
        'STOP':         '-'
}

optimiz_code    = {                                     # one of the accepted keras.optimizers.Optimizer
        'ADAM':         optimizers.Adam,
        'ADAGRAD':      optimizers.Adagrad,
        'SDG':          optimizers.SGD,
        'RMS':          optimizers.RMSprop
}

loss_code       = {                                     # one of the accepted losses functions
        'MSE':          losses.mean_squared_error,
        'BXE':          losses.binary_crossentropy,
        'CXE':          losses.categorical_crossentropy
}

arch_code       = {                                     # one of the accepted architecture classes
        'ENCDEC':       EncoderDecoder
}



def model_summary( model, fname ):
    """ -------------------------------------------------------------------------------------------------
    Print a summary of the model, and plot a graph of the model

    model:          [keras.engine.training.Model]
    fname:          [str] name of the output image with path but without extension
    ------------------------------------------------------------------------------------------------- """
    utils.print_summary( model )
    fname   += '.png'
    utils.plot_model( model, to_file=fname, show_shapes=True, show_layer_names=True )



def create_model( code, arch_kwargs, enc_kwargs, dec_kwargs, summary=False ):
    """ -------------------------------------------------------------------------------------------------
    Create the model

    code:           [str] code of the architecture class
    arch_kwargs:    [dict] parameters of the overall architecture
    enc_kwargs:     [dict] parameters of the encoder part of the architecture
    dec_kwargs:     [dict] parameters of the decoder part of the architecture
    summary:        [str] path where to save plot if you want a summary+plot of model, False otherwise

    return:         the model object
    ------------------------------------------------------------------------------------------------- """
    nn      = arch_code[ code ]( enc_kwargs, dec_kwargs, summary=summary, **arch_kwargs )

    nn.model.compile(
            optimizer       = optimiz_code[ nn.optimiz ]( lr=nn.lrate ),
            loss            = loss_code[ nn.loss_func ],
            loss_weights    = nn.loss_wght if hasattr( nn, 'loss_wght' ) else None
    )

    return nn
