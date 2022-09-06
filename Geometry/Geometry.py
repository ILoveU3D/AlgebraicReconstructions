from abc import ABCMeta, abstractmethod

class Geometry(metaclass=ABCMeta):
    def __init__(self, volumeSize:list, detectorSize:list, H):
        self.volumeSize = volumeSize
        self.detectorSize = detectorSize
        self.H = H

    @abstractmethod
    def fp(self, volume, device):
        pass

    @abstractmethod
    def bp(self, sino, device):
        pass