import sys
sys.path.append('./python')
import numpy as np
import pytest
from needle import backend_ndarray as nd
import needle as ndl
import mugrade
import itertools


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


MATMUL_DIMS = [(16, 16, 16),
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (16, 16, 32),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128)]
@pytest.mark.parametrize("m,n,p", MATMUL_DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_matmul(m, n, p, device):
    _A = np.random.randn(m, n).astype(np.float32)
    _B = np.random.randn(n, p).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    B = ndl.Tensor(nd.array(_B), device=device)
    np.testing.assert_allclose(_A @ _B, (A @ B).numpy(), atol=1e-5, rtol=1e-5)

conv_forward_params = [
    (4, 8, 16, 3, 1),
    (32, 8, 16, 3, 2),
    (32, 8, 8, 3, 2),
    (32, 16, 8, 3, 1),
    (32, 16, 8, 3, 2)
]
@pytest.mark.parametrize("s,cin,cout,k,stride", conv_forward_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_conv_forward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.Conv(cin, cout, k, stride=stride, device=device)
    x = ndl.init.rand(10, cin, s, s, device=device)
    # x = ndl.Tensor(nd.NDArray(np.arange(s*s*cin*1)).reshape((1,cin,s,s)), device=device)

    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy())

    assert np.linalg.norm(f(x).cached_data.numpy() - g(z).data.numpy()) < 1e-3


conv_back_params = [
    (4, 1, 1, 3, 1),
    (14, 8, 16, 3, 1),
    (14, 8, 16, 3, 2),
    (14, 8, 8, 3, 1),
    (14, 8, 8, 3, 2),
    (14, 16, 8, 3, 1),
    (14, 16, 8, 3, 2),
]
@pytest.mark.parametrize("s,cin,cout,k,stride", conv_back_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_conv_backward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.Conv(cin, cout, k, stride=stride, device=device)
    x = ndl.init.rand(1, cin, s, s, device=device, requires_grad=True)

    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy(), requires_grad=True)
    z.requires_grad = True

    res1 = f(x)
    y1 = res1.sum()

    y2 = g(z).sum()

    y1.backward()
    y2.backward()

    assert np.linalg.norm(g.weight.grad.data.numpy() - f.weight.grad.cached_data.numpy().transpose(3, 2, 0, 1)) < 1e-3, "weight gradients match"
    assert np.linalg.norm(g.bias.grad.data.numpy() - f.bias.grad.cached_data.numpy()) < 1e-3, "bias gradients match"
    assert np.linalg.norm(z.grad.data.numpy() - x.grad.cached_data.numpy()) < 1e-3, "input gradients match"


op_conv_shapes = [
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 1, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 1, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 1, 0 ),

    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 2, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 2, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 2, 0 ),

    ( (3, 16, 16, 24), (3, 3, 24, 14), 1, 0 ),
    ( (3, 14, 14, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 1), (5, 5, 1, 16) ,  1, 0),
    ( (3, 17, 17, 16), (5, 5, 16, 1),  1, 0 ),
    ( (3, 17, 17, 16), (1, 1, 16, 1),  1, 0 ),
    ( (1, 14, 14, 2), (3, 3, 2, 2),    1, 0 ),
]
@pytest.mark.parametrize("Z_shape, W_shape, stride, padding", op_conv_shapes)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_op_conv(Z_shape, W_shape, stride, padding, backward, device):
    np.random.seed(0)
    import torch
    _Z = np.random.randn(*Z_shape)*5
    _Z = _Z.astype(np.float32)
    _W = np.random.randn(*W_shape)*5
    _W = _W.astype(np.float32)
    Z = ndl.Tensor(_Z, device=device)
    W = ndl.Tensor(_W, device=device)
    y = ndl.conv(Z, W, padding=padding, stride=stride)
    y2 = y.sum()
    if backward:
        y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True
    Wtch = torch.Tensor(_W).float()
    Wtch.requires_grad=True
    out = torch.nn.functional.conv2d(Ztch.permute(0, 3, 1, 2), Wtch.permute(3, 2, 0, 1), padding=padding, stride=stride)
    out2 = out.sum()
    if backward:
        out2.backward()
    if backward:
        err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
        err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    if backward:
        assert err1 < 1e-2, "input grads match"
        assert err2 < 1e-2, "weight grads match"
    assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)


@pytest.mark.parametrize("device", _DEVICES)
def test_train_cifar10(device):
    np.random.seed(0)
    dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    dataloader = ndl.data.DataLoader(\
             dataset=dataset,
             batch_size=128,
             shuffle=False
             # collate_fn=ndl.data.collate_ndarray,
             # drop_last=False,
             # device=device,
             # dtype="float32"
             )
    from apps.models import ResNet9
    np.random.seed(0)
    model = ResNet9(device=device, dtype="float32")
    out = one_iter_of_cifar10_training(dataloader, model, opt=ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001), device=device)
    assert np.linalg.norm(np.array(list(out)) - np.array([0.09375, 3.5892258])) < 1e-2


def one_iter_of_cifar10_training(dataloader, model, niter=1, loss_fn=ndl.nn.SoftmaxLoss(), opt=None, device=None):
    np.random.seed(4)
    model.train()
    correct, total_loss = 0, 0
    i = 1
    for batch in dataloader:
        opt.reset_grad()
        X, y = batch
        X,y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
        out = model(X)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy() * y.shape[0]
        loss.backward()
        opt.step()
        if i >= niter:
            break
        i += 1
    return correct/(y.shape[0]*niter), total_loss/(y.shape[0]*niter)


######################    |    ######################
###################### MUGRADE ######################
######################    v    ######################

def Prepare(A):
    return (A.numpy().flatten()[:64], A.shape)


def Rand(*shape, device=ndl.cpu(), entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    _A = np.random.randint(low=1, high=10, size=shape)
    return ndl.Tensor(_A, device=device)


def RandC(*shape, entropy=1):
    if ndl.cuda().enabled():
        return Rand(*shape, device=ndl.cuda(), entropy=2)
    else:
        raise NotImplementedError("You need a GPU to run these tests.")


def MugradeSubmit(things):
    mugrade.submit(Prepare(things))
    #print(Prepare(things))


def submit_conv_forward():
    def DoConvOp(batches, cin, cout, n, k=3, stride=1, padding=0, device=ndl.cpu()):
        X = Rand(batches, n, n, cin, device=device)
        W = Rand(k, k, cin, cout, device=device)
        y = ndl.conv(X, W, stride=stride, padding=padding)
        return y

    def DoConvLayer(batches, cin, cout, n, k=3, stride=1, bias=True, device=ndl.cpu()):
        X = Rand(batches, cin, n, n, device=device)
        f = ndl.nn.Conv(cin, cout, k, stride=stride, bias=bias, device=device)
        return f(X)

    MugradeSubmit(DoConvOp(2, 1, 2, 4, k=1, stride=1, padding=0))
    MugradeSubmit(DoConvOp(2, 1, 2, 4, k=1, stride=1, padding=2))
    MugradeSubmit(DoConvOp(2, 3, 1, 6, k=1, stride=2, padding=2))


    MugradeSubmit(DoConvOp(2, 1, 2, 4, k=3, stride=1, padding=0))
    MugradeSubmit(DoConvOp(3, 1, 2, 4, k=3, stride=1, padding=2))
    MugradeSubmit(DoConvOp(1, 1, 3, 6, k=5, stride=2, padding=2))

    MugradeSubmit(DoConvLayer(3, 2, 4, 6, k=3, stride=1, bias=True))
    MugradeSubmit(DoConvLayer(3, 4, 2, 6, k=3, stride=1, bias=True))
    MugradeSubmit(DoConvLayer(1, 1, 1, 12, k=3, stride=2, bias=True))
    MugradeSubmit(DoConvLayer(1, 1, 1, 12, k=1, stride=1, bias=False))
    MugradeSubmit(DoConvLayer(1, 2, 1, 12, k=7, stride=1, bias=False))
    MugradeSubmit(DoConvLayer(1, 1, 3, 12, k=7, stride=4, bias=False))


    if ndl.cuda().enabled():
        MugradeSubmit(DoConvLayer(3, 2, 4, 6, k=3, stride=1, bias=False, device=ndl.cuda()))
        MugradeSubmit(DoConvLayer(3, 4, 2, 6, k=3, stride=1, bias=False, device=ndl.cuda()))
    else:
        print('You need a GPU to run these tests!')


def submit_conv_backward():

    def DoConvOpBackward(batches, cin, cout, n, k=3, stride=1, padding=0, device=ndl.cpu(), wrtX=True):
        X = Rand(batches, n, n, cin, device=device)
        X.requires_grad = True
        W = Rand(k, k, cin, cout, device=device)
        W.requires_grad = True
        y = ndl.conv(X, W, stride=stride, padding=padding).sum()
        y.backward()
        if wrtX:
            return W.grad
        else:
            return X.grad

    def DoConvLayerBackward(batches, cin, cout, n, k=3, stride=1, bias=True, device=ndl.cpu(), wrtX=True):
        X = Rand(batches, cin, n, n, device=device)
        X.requires_grad = True
        f = ndl.nn.Conv(cin, cout, k, stride=stride, bias=bias, device=device)
        y = f(X).sum()
        y.backward()
        if wrtX:
            return f.weight.grad
        else:
            return X.grad

    MugradeSubmit(DoConvOpBackward(2, 1, 2, 4, k=1, stride=1, padding=0, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 6, k=1, stride=2, padding=0, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 1, 2, 10, k=3, stride=1, padding=1, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 8, k=3, stride=2, padding=2, wrtX=True))
    MugradeSubmit(DoConvOpBackward(2, 1, 3, 8, k=5, stride=1, padding=2, wrtX=True))

    MugradeSubmit(DoConvOpBackward(2, 1, 2, 4, k=1, stride=1, padding=0, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 6, k=1, stride=2, padding=0, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 1, 2, 6, k=3, stride=1, padding=1, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 3, 1, 6, k=3, stride=2, padding=2, wrtX=False))
    MugradeSubmit(DoConvOpBackward(2, 1, 3, 8, k=5, stride=1, padding=2, wrtX=False))

    MugradeSubmit(DoConvLayerBackward(3, 2, 4, 6, k=3, stride=1, bias=True, wrtX=True))
    MugradeSubmit(DoConvLayerBackward(1, 2, 1, 12, k=7, stride=1, bias=False, wrtX=True))
    MugradeSubmit(DoConvLayerBackward(1, 1, 3, 12, k=7, stride=4, bias=False, wrtX=True))
    MugradeSubmit(DoConvLayerBackward(3, 2, 4, 6, k=3, stride=1, bias=True, wrtX=False))
    MugradeSubmit(DoConvLayerBackward(1, 2, 1, 12, k=7, stride=1, bias=False, wrtX=False))
    MugradeSubmit(DoConvLayerBackward(1, 1, 3, 12, k=7, stride=4, bias=False, wrtX=False))

    if ndl.cuda().enabled():
        MugradeSubmit(DoConvLayerBackward(3, 2, 4, 6, k=3, stride=1, bias=False, wrtX=True, device=ndl.cuda()))
        MugradeSubmit(DoConvLayerBackward(3, 4, 2, 6, k=3, stride=1, bias=False, wrtX=False, device=ndl.cuda()))
    else:
        print('You need a GPU to run these tests!')


def submit_new_ops():
    # pad
    np.random.seed(1337)
    _A = np.random.randint(low=1, high=10, size=(2, 2, 2, 2))
    A  = nd.NDArray(_A, device=nd.cpu())
    MugradeSubmit(A.pad(( (0, 0), (1, 1), (2, 2), (0, 0))))

    def DoFlip(shape, axes, backward=False, device=ndl.cpu()):
        X = Rand(*shape, device=device)
        X.requires_grad = True
        Y = ndl.flip(X, axes=axes)
        if backward:
            V = Rand(*shape, device=device, entropy=2)
            Z = (V*Y).sum()
            Z.backward()
            return X.grad
        else:
            return Y

    def DoDilate(shape, axes, dilation, backward=False, device=ndl.cpu()):
        X = Rand(*shape, device=device)
        X.requires_grad = True
        Y = ndl.dilate(X, dilation=dilation, axes=axes)
        if backward:
            V = Rand(*Y.shape, device=device, entropy=2)
            Z = (V*Y).sum()
            Z.backward()
            return X.grad
        else:
            return Y

    # flip
    MugradeSubmit(DoFlip((2, 2, 3, 1), (1,2)))
    MugradeSubmit(DoFlip((2, 1, 3, 2), (0,1,2,3)))
    MugradeSubmit(DoFlip((8, 4), (1,)))
    MugradeSubmit(DoFlip((4, 8), (0,)))
    MugradeSubmit(DoFlip((2, 2, 3, 1), (2,3), backward=True))
    MugradeSubmit(DoFlip((2, 1, 3, 2), (1,2,3), backward=True))

    # dilate
    MugradeSubmit(DoDilate((2, 2, 3, 1), (1,2), 1))
    MugradeSubmit(DoDilate((2, 2), (2,), 1))
    MugradeSubmit(DoDilate((2, 2, 3, 1), (1,2), 1, backward=True))
    MugradeSubmit(DoDilate((2, 2), (2,), 1, backward=True))



def submit_resnet9():
    def num_params(model):
        return np.sum([np.prod(x.shape) for x in model.parameters()])

    device = ndl.cpu()
    import sys
    sys.path.append('.')
    from apps.models import ResNet9
    np.random.seed(1)
    model = ResNet9(device=device)

    MugradeSubmit(ndl.Tensor(num_params(model)))

    np.random.seed(1)
    dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    dataloader = ndl.data.DataLoader(\
             dataset=dataset,
             batch_size=128,
             shuffle=True
             )
    np.random.seed(1)
    model = ResNet9(device=device, dtype="float32")
    out = one_iter_of_cifar10_training(dataloader, model, niter=2, opt=ndl.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001), device=device)
    MugradeSubmit(ndl.Tensor(list(out)))


if __name__ == "__main__":
    submit_conv_forward()
    submit_conv_backward()
