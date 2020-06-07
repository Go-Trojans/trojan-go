"""
Please keep the code here for the input features.

"""

import numpy as np

class TrojanGoPlane(Encoder):
	def __init__(self, board_size, plane_size):
		self.board_width, self.board_height = board_size
		self.num_planes = plane_size
