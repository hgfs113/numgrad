import numpy as np
import Matrix


class Dense:
    def __init__(self, in_dim, out_dim):
        self.w = Matrix(np.random.uniform(size=(in_dim, out_dim)), require_grad=True)
        self.b = Matrix(np.random.uniform(size=(out_dim)), require_grad=True)
        self.params = [self.w, self.b]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return x.dot(self.w) + self.b


if __name__ == "__main__":
    X = Matrix(np.random.uniform(size=(50, 10)), require_grad=False)
    y = Matrix(np.random.uniform(size=(50, 1)), require_grad=False)

    layer1 = Dense(10, 10)
    layer2 = Dense(10, 1)
    net_params = layer1.params + layer2.params

    LR = 0.1

    for i in range(10):
        out = layer1(X).relu()
        out = layer2(out)
        loss = MSE(out, y)
        print("loss:", loss.data[0])
        loss.zero_grad()
        loss.backward()
        for param in net_params:
            param.data -= LR * param.grad
