from dataclasses import dataclass
from typing import Any, Callable, Iterable, NewType, Optional, TypeVar, Union

Json = NewType('Json', Any)

T = TypeVar('T')

def find(pred: Callable[[T], bool], items: Iterable[T]) -> Optional[T]:
    return next((x for x in items if pred(x)), None)


@dataclass(frozen=True, order=True)
class Time:
    ms: int

    @classmethod
    def zero(cls) -> 'Time':
        return cls(0)

    @classmethod
    def millis(cls, qty: int) -> 'Time':
        return cls(qty)

    @classmethod
    def seconds(cls, qty: int) -> 'Time':
        return cls(1000 * qty)

    @property
    def s(self) -> int:
        return self.ms // 1000

    @classmethod
    def minutes(cls, qty: int) -> 'Time':
        return cls(60000 * qty)

    @property
    def m(self) -> int:
        return self.ms // 60000

    @classmethod
    def hours(cls, qty: int) -> 'Time':
        return cls(3600000 * qty)

    @property
    def h(self) -> int:
        return self.ms // 3600000

    def __add__(self, other: 'Time') -> 'Time':
        return Time(max(self.ms + other.ms, 0))

    def __sub__(self, other: 'Time') -> 'Time':
        return Time(max(self.ms - other.ms, 0))

    def __mul__(self, scalar: Union[int, float]) -> 'Time':
        return Time(max(round(self.ms * scalar), 0))
