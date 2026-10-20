# micrograd from scratch

a simple autograd engine and neural network library, featuring an implementation of the Adam optimization algorithm and evaluated on the classic non-linear XOR benchmark.

---

## how backpropagation works 
breaking the network down to its individual scalar components demystified the whole process for me. simply put, backprop is a systematic algorithm for applying the chain rule, computing the derivate of the loss in relation to the other parameters within the network. this produces the gradient vector, telling us exactly how each training example wants to nudge weights and biases ($W$ and $b$) to minimize the total loss. 

a logical flow of operations within a network:

* **forward pass**: data flows through the operations (additions, multiplications, activations) to build the computational graph and produce a single scalar loss.
* **local derivatives**: each basic operation only tracks its own local rate of change in isolation (addition passes rates through; multiplication scales by the other input).
* **the starting seed ($\frac{\partial L}{\partial L} = 1.0$)**: all gradients start at `0.0`. backprop primes the root by setting `loss.grad = 1.0`, providing the upstream signal needed for all downstream math.
* **the chain rule (`out.grad`)**: traversing backward in topological order, each node computes its gradient by multiplying its local derivative by the incoming upstream gradient (`out.grad`).
* **accumulation (`+=`)**: when a node’s output branches into multiple paths, its contributions to the loss are summed using `+=` rather than overwritten, keeping to the multivariable chain rule.

---

## graph structure

here is a visualization of the directed acyclic graph (DAG) generated during the forward pass, showing values and their accumulated gradients:

![computation graph]([computation_graph.svg])

---

## benchmark: solving XOR (SGD vs. Adam)

to validate the engine's learning capabilities on a non-linearly separable problem, i trained a 2-layer MLP on the 4-point XOR dataset:

| Input | Target | SGD Prediction | Adam Prediction |
| :---: | :---: | :---: | :---: |
| `[0.0, 0.0]` | `-1.0` | `-0.7582` | `-0.9996` |
| `[0.0, 1.0]` | `1.0` | `-0.0365` | `0.9995` |
| `[1.0, 0.0]` | `1.0` | `0.7280` | `0.9887` |
| `[1.0, 1.0]` | `-1.0` | `0.0343` | `-0.9907` |

> under the same number of epochs, standard SGD struggled to separate the middle states (predicting `-0.0365` and `0.0343`), while Adam cleanly converged to nearly $\pm 1.0$ across all four truth table conditions.

### training loss curves

![XOR training curve: SGD vs Adam]([xor_loss_curve.png])


