import numpy as np


class Matrix:
    data: np.ndarray = None
    grad: np.ndarray = None
    require_grad: bool = False
    _backward = lambda _: None
    children: set = set()

    def __init__(self, data, require_grad=True, children=set()):
        self.data = data
        self.require_grad = require_grad
        if self.require_grad:
            self.grad = np.zeros_like(data)
        self.children = children

    def _broadcast_grad(self, _in, _out):
        if len(_in.shape) != len(_out.shape):
            for i in range(len(_out.shape) - len(_in.shape)):
                _out = _out.mean(0)
        return _out

    def __repr__(self):
        return f"data: {self.data},\ngrad: {self.grad}"

    def __add__(self, other):
        out = Matrix(self.data + other.data, children=(self, other))

        def _backward():
            if self.require_grad:
                self.grad += self._broadcast_grad(self.grad, out.grad)
            if other.require_grad:
                other.grad += self._broadcast_grad(other.grad, out.grad)

        out._backward = _backward
        return out

    def __pow__(self, deg):
        out = Matrix(self.data ** deg, children=(self, ))

        if self.require_grad:
            def _backward():
                self.grad += deg * out.grad * self.data ** (deg - 1)
            out._backward = _backward
        return out

    def dot(self, other):
        out = Matrix(self.data @ other.data, children=(self, other))

        def _backward():
            if self.require_grad:
                self.grad += out.grad @ other.data.T
            if other.require_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self):
        mask = self.data > 0
        out = Matrix(np.where(mask, self.data, 0), children=(self,))

        if self.require_grad:
            def _backward():
                self.grad = out.grad * mask

            out._backward = _backward
        return out

    def mean(self, dim=0):
        assert isinstance(dim, int)

        out = Matrix(self.data.mean(dim), children=(self, ))

        if self.require_grad:
            def _backward():
                self.grad = np.ones_like(self.data) / self.data.shape[dim] * np.expand_dims(out.grad, dim)

            out._backward = _backward
        return out

    def sum(self, dim=0):
        out = Matrix(self.data.sum(dim), children=(self, ))

        if self.require_grad:
            def _backward():
                self.grad = np.ones_like(self.data) * np.expand_dims(out.grad, dim)

            out._backward = _backward
        return out

    def _build_graph(self):
        visited = set()
        nodes = []

        def _visit(node, nodes):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    if child.require_grad:
                        _visit(child, nodes)
                nodes.append(node)
        _visit(self, nodes)
        return nodes

    def zero_grad(self):
        nodes = self._build_graph()
        for node in nodes:
            node.grad *= 0 # faster than np.zeros?

    def backward(self):
        assert self.require_grad
        nodes = self._build_graph()
        self.grad = np.ones_like(self.data)
        for node in reversed(nodes):
            node._backward()

    def __neg__(self):
        out = Matrix(-self.data, children=(self, ))
        if self.require_grad:
            def _backward():
                self.grad += -out.grad

            out._backward = _backward
        return out

    def __mul__(self, other):
        out = Matrix(self.data * other.data, children=(self, other))

        def _backward():
            if self.require_grad:
                self.grad += other.data * out.grad
            if other.require_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1


if __name__ == "__main__":
    X = Matrix(np.random.uniform(size=(10, 4)), require_grad=False)
    y = Matrix(np.random.uniform(size=(10, 1)), require_grad=False)
    w = Matrix(np.random.uniform(size=(4, 1)))
    b = Matrix(np.random.uniform(size=(1)))
    params = [w, b]

    for i in range(10):
        loss = ((X.dot(w) + b - y) ** 2).mean(0)
        print("loss:", loss.data[0])
        loss.zero_grad()
        loss.backward()
        for param in params:
            param.data -= 0.1 * param.grad
