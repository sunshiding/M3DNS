from torch.autograd import Function
from torch import nn

class WassersteinLoss(Function):
    def __init__(self, k, reg):
        super(WassersteinLoss, self).__init__()
        self.m = np.zeros(k.shape)
        self.m = self.update_cost_matrix(k)
        self.log = {}
        self.reg = reg
    def get_m(self):
        return self.m

    def update_m(a):
        self.m = a

    def update_cost_matrix(self,k):
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                self.m[i][j] = k[i][i] + k[j][j] - 2*k[i][j]

    def forward(self, x, y):
        self.save_for_backward(x, y)
        loss = torch.zeros(1).cuda()
        for i in range(x.shape[0]):
            a = x[i].cpu().numpy()
            a[a <= 0] = 1e-9
            a = a / np.sum(a)
            #print(y[i])
            b = y[i].cpu().numpy()
            b[b <= 0] = 1e-9
            b = b / np.sum(b)

            dis, log_i = ot.sinkhorn2(a, b, self.m, self.reg, log=True)
            self.log[i] = torch.FloatTensor(log_i['u'])
            loss += dis[0]

        return loss

    def backward(self, grad_output):
        x, y, = self.saved_tensors

        grad_x = []
        L = y.shape[1]
        e = torch.ones(1, x.shape[1]).cuda()
        for i in range(x.shape[0]):
            u = self.log[i].cuda()
            u = torch.log(u.view(1, -1)) / self.reg - torch.log(torch.sum(u, 0))[0] * e / (self.reg * L)
            grad_x.append(u)

        grad_x = torch.cat(grad_x)
        return grad_x, None