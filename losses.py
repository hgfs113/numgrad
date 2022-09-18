import numpy as np


def MSELoss:
    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        return ((x - y) ** 2).mean(0)


class BCELogitsLoss:
    def __call__(self, logits, labels):
        return self.forward(logits, labels)

    def forward(self, logits, labels):
        out = Matrix(np.logaddexp(0, (2 * y.data - 1) * logit.data), require_grad=True, children=(logit, ))

        def _backward():
            logit.grad += logit.data * np.exp(-out.data) * out.grad

        out._backward = _backward
        return out.mean(0)
