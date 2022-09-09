import os
import yaml
import torch
from Geometry.Geometry import Geometry
from ..Algorithm import Algorithm
from .l1ball import l1ball
from .hyperParams import power_method
from .gradients import spraseMatrixZ, spraseMatrixY, spraseMatrixX, spraseMatrixI, getNormK

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)
class DTVCP(Algorithm):
    def __init__(self, geometry:Geometry):
        super(DTVCP, self).__init__(geometry)
        self.cascades = params["epoch"]
        self.debug = params["debug"]
        self.device = params["device"]
        self.feqs = params["debugfeqs"]
        self.dx, self.dxt, normDx = spraseMatrixX(tuple(geometry.volumeSize), self.device)
        self.dy, self.dyt, normDy = spraseMatrixY(tuple(geometry.volumeSize), self.device)
        self.dz, self.dzt, normDz = spraseMatrixZ(tuple(geometry.volumeSize), self.device)
        _, _, normI = spraseMatrixI(tuple(geometry.volumeSize), self.device)
        normH = power_method(geometry.H)
        self.v1 = normH / normDx
        self.v2 = normH / normDy
        self.v3 = normH / normDz
        self.mu = normH / normI
        L = getNormK(geometry.H, [1, self.v1, self.v2, self.v3, self.mu])
        print("||K||={}".format(L))
        self.tao = 1 / L
        self.sigma = 1 / L

    def run(self, image, sino):
        image = image.to(self.device)
        sino = sino.to(self.device)
        w = torch.zeros_like(sino)
        p = q = s = c = torch.zeros_like(image)
        f = f_ = image
        for cascade in range(self.cascades):
            if cascade % 10 == 0:
                self.tx = torch.sum(torch.abs(self._getGradient(f, self.dx))).item()
                self.ty = torch.sum(torch.abs(self._getGradient(f, self.dy))).item()
                self.tz = torch.sum(torch.abs(self._getGradient(f, self.dz))).item()
            res = self.geometry.fp(f_, self.device) - sino
            w = (w + self.sigma * res) / (1 + self.sigma)
            recon = self.geometry.bp(w, self.device)
            p_ = p + self.v1 * self.sigma * self._getGradient(f_, self.dx)
            q_ = q + self.v2 * self.sigma * self._getGradient(f_, self.dy)
            s_ = s + self.v3 * self.sigma * self._getGradient(f_, self.dz)
            p = p_ - self.sigma * torch.sign(p_) * l1ball(torch.abs(p_) / self.sigma, self.v1 * self.tx)
            q = q_ - self.sigma * torch.sign(q_) * l1ball(torch.abs(q_) / self.sigma, self.v2 * self.ty)
            s = s_ - self.sigma * torch.sign(s_) * l1ball(torch.abs(s_) / self.sigma, self.v3 * self.tz)
            c = -torch.nn.functional.relu(-c - self.sigma * self.mu * f_)
            p_ = self.v1 * self._getGradient(p, self.dxt)
            q_ = self.v2 * self._getGradient(q, self.dyt)
            s_ = self.v3 * self._getGradient(s, self.dzt)
            c_ = self.mu * c
            fnew = f - recon - self.tao * (p_ + q_ + s_ + c_)
            f_ = 2 * fnew - f
            f = fnew
            loss = torch.sum(torch.abs(res))
            print("{}: loss is {:.2f}".format(cascade, loss))
            if self.debug is not None and cascade % self.feqs == 0:
                f.detach().cpu().numpy().tofile(os.path.join(self.debug, "result_{}_{}.raw".format(cascade, loss)))
        return f

    def _getGradient(self, image, sparse):
        return torch.reshape(torch.matmul(sparse, image.view(-1)), image.size())
