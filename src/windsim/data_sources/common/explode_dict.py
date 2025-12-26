from itertools import product
from collections.abc import Mapping


type OneOrMany[T] = T | list[T] | tuple[T] | set[T] | frozenset[T]
_Many = list | tuple | set | frozenset


class ProductDimension[T](tuple[T, ...]):
    """Explicitly marks a sequence to be exploded via Cartesian product."""
    pass


def each[T](*args: T) -> ProductDimension[T]:
    return ProductDimension(args)


type Explodable[T] = T | ProductDimension[T]


def explode[T](request: Mapping[str, Explodable[T]]) -> list[dict[str, T]]:
    def as_product_dimension(values: Explodable[T]) -> ProductDimension[T]:
        if isinstance(values, ProductDimension):
            return ProductDimension(values)
        return ProductDimension([values])

    dimensions = (as_product_dimension(values) for values in request.values())
    exploded_requests = [dict(zip(request.keys(), values)) for values in product(*dimensions)]
    return exploded_requests
