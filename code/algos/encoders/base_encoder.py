"""
This will be the base encoder class definition which will be generic and can be used to indentify different
encoders for different board sizes.
"""

import importlib

class Encoder:
    def name(self):  # <1>
        raise NotImplementedError()



# <1> Lets us support logging or saving the name of the encoder our model is using.
