from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    pos_x, neg_x = list(vals), list(vals)
    pos_x[arg] += epsilon
    neg_x[arg] -= epsilon
    pos_f = f(*pos_x)
    neg_f = f(*neg_x)
    return (pos_f - neg_f) / 2 / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    variable_list = []
    visited = {}

    def _visit(var: Variable):
        if var.is_constant():
            return
        if var.unique_id in visited:
            return
        if not var.is_leaf():
            for input in var.parents:
                _visit(input)
        visited[var.unique_id] = var.unique_id
        variable_list.append(var)

    _visit(variable)
    variable_list.reverse()
    return variable_list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    variable_list = topological_sort(variable)
    var_der_dict = {variable.unique_id: deriv}
    for var in variable_list:
        if var.is_leaf():
            continue
        if var.unique_id in var_der_dict.keys():
            deriv = var_der_dict[var.unique_id]
        deriv_dict = var.chain_rule(deriv)
        for var_, d in deriv_dict:
            if var_.is_leaf():
                var_.accumulate_derivative(d)
            if var_.unique_id in var_der_dict.keys():
                var_der_dict[var_.unique_id] += d
            else:
                var_der_dict[var_.unique_id] = d




@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
