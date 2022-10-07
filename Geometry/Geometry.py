from abc import ABCMeta, abstractmethod

class Geometry(metaclass=ABCMeta):
    def __init__(self, volumeSize, detectorSize):
        self.volumeSize = volumeSize
        self.detectorSize = detectorSize

    @abstractmethod
    def fp(self, volume, device):
        pass

    @abstractmethod
    def bp(self, sino, device):
        pass