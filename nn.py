import numpy as np
import Matrix
import losses


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
    def test_loss(loss_fn, data_size=50, hid_dim=20, clf=False):
        X = Matrix(np.random.uniform(size=(data_size, hid_dim)), require_grad=False)
        if clf:  # classification
            y = Matrix(np.random.randint(2, size=(data_size, 1)), require_grad=False)
        else: # regression
            y = Matrix(np.random.uniform(size=(data_size, 1)), require_grad=False)

        layer1 = Dense(hid_dim, hid_dim)
        layer2 = Dense(hid_dim, 1)
        net_params = layer1.params + layer2.params

        LR = 0.1

        for i in range(10):
            out = layer1(X).relu()
            out = layer2(out)
            loss = loss_fn(out, y)
            print("loss:", loss.data[0])
            loss.zero_grad()
            loss.backward()
            for param in net_params:
                param.data -= LR * param.grad

    print("MSELoss:")
    test_loss(losses.MSELoss)
    print()
    print("BCELoss:")
    test_loss(losses.BCELogitsLoss)
