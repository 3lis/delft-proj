"""
#####################################################################################################################

    Delft visiting project
    Alice   2020

    Configuration file

#####################################################################################################################
"""

kwargs      = {
        # -------- GENERAL ---------------------------------------- #
        'n_epochs':         100,                                    # [int] number of epochs
        'batch_size':       32,                                     # [int] size of batch
        'arch_code':        'ENCDEC',                               # [str] code of model architecture
        'input_class':      'ATT',                                  # [str] code of input class
        'target_class':     'WRP',                                  # [str] code of target class
        'dataset_seq':      50,                                     # [int] number of used sequences in dataset
        'black_list':       True,                                   # [bool] if True remove bad sequences in dataset
        'process_workers':  200,                                    # [int] max number of processes

        # -------- ARCHITECTURE ----------------------------------- #
        'arch_kwargs':      {
                'arch_layout':          'CPCPCPCPFD-DRTTT',         # [str] code describing the order of layers in the model
                'input_size':           ( 900, 1600, 3 ),           # [tuple] height, width, channels
                'lrate':                1e-4,                       # [float] learning rate
                'optimiz':              'ADAM',                     # [str] code of the optimizer
                'loss_func':            'BXE'                       # [str] code of the loss function
        },
        # -------- ENCODER ---------------------------------------- #
        'enc_kwargs':       {
                'conv_kernel_num':      [ 64, 64, 64, 64 ],         # [list of int] number of kernels for each convolution
                'conv_kernel_size':     [ 5, 5, 3, 3 ],             # [list of int] (square) size of kernels for each convolution
                'conv_strides':         [ 2, 2, 1, 1 ],             # [list of int] stride for each convolution
                'pool_size':            [ 4, 4, 2, 2 ],             # [list of int] pooling size for each MaxPooling
                'enc_dnse_size':        [ 512 ]                     # [list of int] size of each dense layer
        },
        # -------- DECODER ---------------------------------------- #
        'dec_kwargs':       {
                'first_3d_shape':       ( 4, 4, 128 ),              # [tuple] the first 3D shape after reshaping
                'dcnv_kernel_num':      [ 64, 32, 1 ],              # [list of int] number of kernels for each deconvolution
                'dcnv_kernel_size':     [ 3, 5, 5 ],                # [list of int] (square) size of kernels for each deconv
                'dcnv_strides':         [ 2, 4, 4 ],                # [list of int] stride for each deconvolution
                'dec_dnse_size':        [ 2048 ]                    # [list of int] size of each dense layer
        }
}
