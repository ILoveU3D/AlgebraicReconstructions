from abc import ABCMeta, abstractmethod
from Geometry.Geometry import Geometry

class Algorithm(metaclass=ABCMeta):
    def __init__(self, geometry:Geometry):
        self.geometry = geometry

    @abstractmethod
    def run(self, image, sino):
        pass