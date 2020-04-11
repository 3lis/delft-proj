"""
#####################################################################################################################

    Delft visiting project
    Alice   2020

    Configuration class, checking all the parameters used in the software

#####################################################################################################################
"""

import  os
from    argparse    import ArgumentParser

from    data_gen    import class_code
from    arch_net    import arch_code, layer_code, optimiz_code, loss_code


class Config( object ):
    """ -------------------------------------------------------------------------------------------------------------
    LIST of all PARAMETERS accepted by the software (* indicates compulsory parameters)

    CONFIG:               * [str] name of configuration file (without path nor extension)
    FGPU:                   [float] fraction of GPU memory to allocate (DEFAULT=0.9)
    GPU:                  * [int or list of int] number of GPUs to use (0=CPU) or list of GPU indices
    LOAD:                   [str] pathname of HDF5 file to load as weights
    REDIRECT:               [bool] redirect stderr and stdout to log files (DEFAULT=False)
    ARCHIVE:                [bool] archive python scripts (DEFAULT=False)
    TRAIN:                  [bool] execute training of the model (DEFAULT=False)
    TEST:                   [bool] execute testing of the model (DEFAULT=False)

    n_epochs:             * [int] number of epochs
    batch_size:           * [int] batch size
    arch_code:            * [str] code of the architecture class (one of 'arch_code' in arch_net.py)
    input_class:          * [str] class code of input data (one of 'class_code' in data_gen.py)
    target_class:         * [str] class code of target data  (one of 'class_code' in data_gen.py)

    black_list:             [bool] use pre-computed black list to remove bad sequences from dataset (DEFAULT=False)
    dataset_seq:            [int] max number of sequences to use in the dataset (if None use all) (DEFAULT=None)
    shuffle_batch_samples:  [bool] at each epoch, randomly distribute the samples in the batches (DEAFULT=True)
    shuffle_batch_order:    [bool] at the beginning of each epoch, shuffle the order of the batches (DEAFULT=True)
    shuffle_partitions:     [bool] randomly partition the dataset in train/valid/test (DEAFULT=True)
    process_queue:          [int] max size for the generator queue (DEAFULT=10)
    process_multi:          [bool] use process-based threading (DEAFULT=False)
    process_workers:        [int] max number of processes (DEAFULT=1)
    chckpnt:                [int] interval (number of epochs) between saving checkpoints (DEFAULT=0)

    arch_kwargs:          * [dict] general parameters of architecture, containing:
        arch_layout:          * [str] code describing the order of layers in the model (using 'layer_code' in arch_net.py)
        optimiz:              * [str] code of the optimizer (one of 'optimiz_code' in arch_net.py)
        loss:                 * [str] code of the loss function (one of 'loss_code' in arch_net.py)
        lrate:                * [float] learning rate
        input_size:             [list] height, width, channels (DEFAULT is taken from 'class_code' in data_gen.py)
        output_size:            [list] height, width, channels (DEFAULT is taken from 'class_code' in data_gen.py)

        kl_weight:            * [float] weight of KL component in loss function (ONLY FOR VARIATIONAL MODELS)

    enc_kwargs:             [dict] parameters of encoder network, containing:
        conv_kernel_num:      * [list of int] number of kernels for each convolution
        conv_kernel_size:     * [list of int] (square) size of kernels for each convolution
        conv_strides:         * [list of int] stride for each convolution
        conv_padding:           [list of str] padding (same/valid) for each convolution (DEFAULT=same)
        conv_activation:        [list of str] activation function for each convolution (DEFAULT=relu)
        conv_train:             [list of bool] False to lock training of each convolution (DEFAULT=True)
        pool_size:              [list of int] pooling size for each max-pooling
        enc_dnse_size:        * [list of int] size of each dense layer
        enc_dnse_dropout:       [list of int] dropout of each dense layer (DEFAULT=0)
        enc_dnse_activation:    [list of str] activation function for each dense layer (DEFAULT=relu)
        enc_dnse_train:         [list of bool] False to lock training of each dense layer (DEFAULT=True)

    dec_kwargs:             [dict] parameters of encoder network, containing:
        first_3d_shape:       * [list] the first 3D shape after reshaping (height, width, channels)
        dcnv_kernel_num:      * [list of int] number of kernels for each deconvolution
        dcnv_kernel_size:     * [list of int] (square) size of kernels for each deconvolution
        dcnv_strides:         * [list of int] stride for each deconvolution
        dcnv_padding:           [list of str] padding (same/valid) for each deconvolution (DEFAULT=same)
        dcnv_activation:        [list of str] activation function for each deconvolution (DEFAULT=relu)
        dcnv_train:             [list of bool] False to lock training of each deconvolution (DEFAULT=True)
        dec_dnse_size:        * [list of int] size of each dense layer
        dec_dnse_dropout:       [list of int] dropout of each dense layer (DEFAULT=0)
        dec_dnse_activation:    [list of str] activation function for each dense layer (DEFAULT=relu)
        dec_dnse_train:         [list of bool] False to lock training of each dense layer (DEFAULT=True)
    ------------------------------------------------------------------------------------------------------------- """

    def load_from_line( self, **line_kwargs ):
        """ ---------------------------------------------------------------------------------------------------------
        Load parameters from arguments passed in command line. Check the existence and correctness of all
        required parameteres

        line_kwargs:        [dict] parameteres read from arguments passed in command line
        --------------------------------------------------------------------------------------------------------- """
        for key, value in line_kwargs.items():
            setattr( self, key, value )

        assert os.path.isfile( 'src/cnfg/' + self.CONFIG + '.py' ), "Configuration file must be located in 'src/cnfg'"

        if self.LOAD is not None:
            assert os.path.isfile( self.LOAD ), "File '{}' not found".format( self.LOAD )

        assert self.FGPU <= 1.0



    def load_from_file( self, **file_kwargs ):
        """ ---------------------------------------------------------------------------------------------------------
        Load parameters from a python file. Check the existence and correctness of all required parameteres

        In some cases, if a parameter is not passed as argument, it is set to a default value.
        In other cases, if a parameter is considered fundamental, it must be specified as argument.

        file_kwargs:        [dict] parameteres coming from a python module (file)
        --------------------------------------------------------------------------------------------------------- """
        for key, value in file_kwargs.items():
            setattr( self, key, value )


        # -------- GENERAL -------- 
        assert hasattr( self, 'n_epochs' )                  and isinstance( self.n_epochs, int )
        assert hasattr( self, 'batch_size' )                and isinstance( self.batch_size, int )
        assert hasattr( self, 'arch_code' )                 and self.arch_code in arch_code

        # TODO implement multi-class
        # TODO implement multi-class
        assert hasattr( self, 'input_class' )               and self.input_class in class_code
        assert hasattr( self, 'target_class' )              and self.target_class in class_code

        if not hasattr( self, 'black_list' ):               self.black_list             = False
        if not hasattr( self, 'dataset_seq' ):              self.dataset_seq            = None
        if not hasattr( self, 'shuffle_batch_samples' ):    self.shuffle_batch_samples  = True
        if not hasattr( self, 'shuffle_batch_order' ):      self.shuffle_batch_order    = True
        if not hasattr( self, 'shuffle_partitions' ):       self.shuffle_partitions     = True
        if not hasattr( self, 'process_queue' ):            self.process_queue          = 10
        if not hasattr( self, 'process_multi' ):            self.process_multi          = False
        if not hasattr( self, 'process_workers' ):          self.process_workers        = 1
        if not hasattr( self, 'chckpnt' ):                  self.chckpnt                = 0


        # -------- ARCHITECTURE -------- 
        assert hasattr( self, 'arch_kwargs' )               and isinstance( self.arch_kwargs, dict )

        assert 'arch_layout' in self.arch_kwargs            and isinstance( self.arch_kwargs[ 'arch_layout' ], str )
        assert 'optimiz' in self.arch_kwargs                and self.arch_kwargs[ 'optimiz' ] in optimiz_code
        assert 'loss' in self.arch_kwargs                   and self.arch_kwargs[ 'loss' ] in loss_code
        assert 'lrate' in self.arch_kwargs                  and isinstance( self.arch_kwargs[ 'lrate' ], float )

        if 'input_size' not in self.arch_kwargs:
            self.arch_kwargs[ 'input_size' ]                = class_code[ self.input_class ][ 'size' ]
        if 'output_size' not in self.arch_kwargs:
            self.arch_kwargs[ 'output_size' ]               = class_code[ self.target_class ][ 'size' ]

        # case of VARIATIONAL model
        if self.arch_code == 'VAR-ENCDEC':
            assert 'kl_weight' in self.arch_kwargs          and isinstance( self.arch_kwargs[ 'kl_weight' ], float )
            # TODO implement KL_increment


        # case of model with ENCODERs and DECODERs
        if self.arch_code in ( 'ENCDEC', 'VAR-ENCDEC' ):
            # TODO implement multi-enc
            # TODO implement multi-dec
            assert hasattr( self, 'enc_kwargs' )                and isinstance( self.arch_kwargs, dict )    
            assert hasattr( self, 'dec_kwargs' )                and isinstance( self.arch_kwargs, dict )

            # -------- ENCODER -------- 
            n_conv      =  self.arch_kwargs[ 'arch_layout' ].split( '-' )[ 0 ].count( layer_code[ 'CONV' ] )
            n_pool      =  self.arch_kwargs[ 'arch_layout' ].split( '-' )[ 0 ].count( layer_code[ 'POOL' ] )
            n_dnse      =  self.arch_kwargs[ 'arch_layout' ].split( '-' )[ 0 ].count( layer_code[ 'DNSE' ] )

            assert 'conv_kernel_num' in self.enc_kwargs         and len( self.enc_kwargs[ 'conv_kernel_num' ] ) == n_conv
            assert 'conv_kernel_size' in self.enc_kwargs        and len( self.enc_kwargs[ 'conv_kernel_size' ] ) == n_conv
            assert 'conv_strides' in self.enc_kwargs            and len( self.enc_kwargs[ 'conv_strides' ] ) == n_conv
            assert 'enc_dnse_size' in self.enc_kwargs           and len( self.enc_kwargs[ 'enc_dnse_size' ] ) == n_dnse

            if n_pool > 0:
                assert 'pool_size' in self.enc_kwargs           and len( self.enc_kwargs[ 'pool_size' ] ) == n_pool

            if 'conv_padding' not in self.enc_kwargs:
                self.enc_kwargs[ 'conv_padding' ]               = n_conv * [ 'same' ]
            elif isinstance( self.enc_kwargs[ 'conv_padding' ], str ):
                self.enc_kwargs[ 'conv_padding' ]               = n_conv * [ self.enc_kwargs[ 'conv_padding' ] ]

            if 'conv_activation' not in self.enc_kwargs:
                self.enc_kwargs[ 'conv_activation' ]            = n_conv * [ 'relu' ]
            elif isinstance( self.enc_kwargs[ 'conv_activation' ], str ):
                self.enc_kwargs[ 'conv_activation' ]            = n_conv * [ self.enc_kwargs[ 'conv_activation' ] ]

            if 'conv_train' not in self.enc_kwargs:
                self.enc_kwargs[ 'conv_train' ]                 = n_conv * [ 'True' ]
            elif isinstance( self.enc_kwargs[ 'conv_train' ], str ):
                self.enc_kwargs[ 'conv_train' ]                 = n_conv * [ self.enc_kwargs[ 'conv_train' ] ]

            if 'enc_dnse_dropout' not in self.enc_kwargs:
                self.enc_kwargs[ 'enc_dnse_dropout' ]           = n_dnse * [ 0 ]
            else:
                assert len( self.enc_kwargs[ 'enc_dnse_dropout' ] ) == n_dnse

            if 'enc_dnse_activation' not in self.enc_kwargs:
                self.enc_kwargs[ 'enc_dnse_activation' ]        = n_dnse * [ 'relu' ]
            elif isinstance( self.enc_kwargs[ 'enc_dnse_activation' ], str ):
                self.enc_kwargs[ 'enc_dnse_activation' ]        = n_dnse * [ self.enc_kwargs[ 'enc_dnse_activation' ] ]

            if 'enc_dnse_train' not in self.enc_kwargs:
                self.enc_kwargs[ 'enc_dnse_train' ]             = n_dnse * [ 'True' ]
            elif isinstance( self.enc_kwargs[ 'enc_dnse_train' ], str ):
                self.enc_kwargs[ 'enc_dnse_train' ]             = n_dnse * [ self.enc_kwargs[ 'enc_dnse_train' ] ]


            # -------- DECODER -------- 
            n_dcnv      =  self.arch_kwargs[ 'arch_layout' ].split( '-' )[ -1 ].count( layer_code[ 'DCNV' ] )
            n_dnse      =  self.arch_kwargs[ 'arch_layout' ].split( '-' )[ -1 ].count( layer_code[ 'DNSE' ] )

            assert 'dcnv_kernel_num' in self.dec_kwargs         and len( self.dec_kwargs[ 'dcnv_kernel_num' ] ) == n_dcnv
            assert 'dcnv_kernel_size' in self.dec_kwargs        and len( self.dec_kwargs[ 'dcnv_kernel_size' ] ) == n_dcnv
            assert 'dcnv_strides' in self.dec_kwargs            and len( self.dec_kwargs[ 'dcnv_strides' ] ) == n_dcnv
            assert 'dec_dnse_size' in self.dec_kwargs           and len( self.dec_kwargs[ 'dec_dnse_size' ] ) == n_dnse
            assert 'first_3d_shape' in self.dec_kwargs          and len( self.dec_kwargs[ 'first_3d_shape' ] ) == 3

            if 'dcnv_padding' not in self.dec_kwargs:
                self.dec_kwargs[ 'dcnv_padding' ]               = n_dcnv * [ 'same' ]
            elif isinstance( self.dec_kwargs[ 'dcnv_padding' ], str ):
                self.dec_kwargs[ 'dcnv_padding' ]               = n_dcnv * [ self.dec_kwargs[ 'dcnv_padding' ] ]

            if 'dcnv_activation' not in self.dec_kwargs:
                self.dec_kwargs[ 'dcnv_activation' ]            = n_dcnv * [ 'relu' ]
            elif isinstance( self.dec_kwargs[ 'dcnv_activation' ], str ):
                self.dec_kwargs[ 'dcnv_activation' ]            = n_dcnv * [ self.dec_kwargs[ 'dcnv_activation' ] ]

            if 'dcnv_train' not in self.dec_kwargs:
                self.dec_kwargs[ 'dcnv_train' ]                 = n_dcnv * [ 'True' ]
            elif isinstance( self.dec_kwargs[ 'dcnv_train' ], str ):
                self.dec_kwargs[ 'dcnv_train' ]                 = n_dcnv * [ self.dec_kwargs[ 'dcnv_train' ] ]

            if 'dec_dnse_dropout' not in self.dec_kwargs:
                self.dec_kwargs[ 'dec_dnse_dropout' ]           = n_dnse * [ 0 ]
            else:
                assert len( self.dec_kwargs[ 'dec_dnse_dropout' ] ) == n_dnse

            if 'dec_dnse_activation' not in self.dec_kwargs:
                self.dec_kwargs[ 'dec_dnse_activation' ]        = n_dnse * [ 'relu' ]
            elif isinstance( self.dec_kwargs[ 'dec_dnse_activation' ], str ):
                self.dec_kwargs[ 'dec_dnse_activation' ]        = n_dnse * [ self.dec_kwargs[ 'dec_dnse_activation' ] ]

            if 'dec_dnse_train' not in self.dec_kwargs:
                self.dec_kwargs[ 'dec_dnse_train' ]             = n_dnse * [ 'True' ]
            elif isinstance( self.dec_kwargs[ 'dec_dnse_train' ], str ):
                self.dec_kwargs[ 'dec_dnse_train' ]             = n_dnse * [ self.dec_kwargs[ 'dec_dnse_train' ] ]



    def __str__( self ):
        """ ---------------------------------------------------------------------------------------------------------
        Visualize the list of all parameters
        --------------------------------------------------------------------------------------------------------- """
        s   = ''
        d   = self.__dict__

        for k in d:
            if isinstance( d[ k ], dict ):
                s   += "{}:\n".format( k )
                for j in d[ k ]:
                    s   += "{:5}{:<30}{}\n".format( '', j, d[ k ][ j ] )
            else:
                s   += "{:<35}{}\n".format( k, d[ k ] )

        return s


# ===================================================================================================================


def read_args():
    """ -------------------------------------------------------------------------------------------------------------
    Parse the command-line arguments defined by flags
    
    return:         [dict] key = name of parameter, value = value of parameter
    ------------------------------------------------------------------------------------------------------------- """
    parser      = ArgumentParser()

    parser.add_argument(
            '-c',
            '--config',
            action          = 'store',
            dest            = 'CONFIG',
            type            = str,
            required        = True,
            help            = "Name of configuration file (without path nor extension)"
    )
    parser.add_argument(
            '-f',
            '--fgpu',
            action          = 'store',
            dest            = 'FGPU',
            type            = float,
            default         = 0.90,
            help            = "Fraction of GPU memory to allocate"
    )
    parser.add_argument(
            '-g',
            '--gpu',
            action          = 'store',
            dest            = 'GPU',
            required        = True,
            help            = "Number of GPUs to use (0 if CPU) or list of GPU indices"
    )
    parser.add_argument(
            '-l',
            '--load',
            action          = 'store',
            dest            = 'LOAD',
            type            = str,
            default         = None,
            help            = "HDF5 file to load as weights or entire model"
    )
    parser.add_argument(
            '-r',
            '--redir',
            action          = 'store_true',
            dest            = 'REDIRECT',
            help            = "Redirect stderr and stdout to log files"
    )
    parser.add_argument(
            '-s',
            '--save',
            action          = 'store_true',
            dest            = 'ARCHIVE',
            help            = "Archive python scripts"
    )
    parser.add_argument(
            '-T',
            '--train',
            action          = 'store_true',
            dest            = 'TRAIN',
            help            = "Execute training of the model"
    )
    parser.add_argument(
            '-t',
            '--test',
            action          = 'store_true',
            dest            = 'TEST',
            help            = "Execute testing of the model"
    )

    return vars( parser.parse_args() )
