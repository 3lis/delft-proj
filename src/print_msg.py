"""
#####################################################################################################################

    Delft visiting project
    Alice   2020

    Utilities for printing messages

#####################################################################################################################
"""

import      os
import      sys
import      inspect


def print_err( msg, exit=True ):
    """ ---------------------------------------------------------------------------------------------------------
    Print an error message, including the file and line number where the print is called

    msg:        [str] message to print
    exit:       [bool] if True exit programme
    --------------------------------------------------------------------------------------------------------- """
    LINE    = inspect.currentframe().f_back.f_lineno
    FILE    = os.path.basename( inspect.getfile( inspect.currentframe().f_back ) )

    print( "ERROR [{}:{}] --> {}\n".format( FILE, LINE, msg ) )

    if exit:
        sys.exit( 1 )



def print_wrn( msg ):
    """ ---------------------------------------------------------------------------------------------------------
    Print a warning message, including the file and line number where the print is called

    msg:        [str] message to print
    --------------------------------------------------------------------------------------------------------- """
    LINE    = inspect.currentframe().f_back.f_lineno
    FILE    = os.path.basename( inspect.getfile( inspect.currentframe().f_back ) )

    print( "WARNING [{}:{}] --> {}\n".format( FILE, LINE, msg ) )
