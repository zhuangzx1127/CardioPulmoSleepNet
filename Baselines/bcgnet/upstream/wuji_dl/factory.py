import importlib
import typing as t


_T = t.TypeVar("_T")


class Factory:
    REGISTER: dict[str, t.Any] = {}

    @classmethod
    def register(cls, key: str) -> t.Callable[[_T], _T]:
        def wrapper(val: _T) -> _T:
            cls.REGISTER[key] = val
            return val

        return wrapper


def get_class(base: t.Type[_T], name: str) -> t.Type[_T]:
    if issubclass(base, Factory) and name in base.REGISTER:
        cls = base.REGISTER[name]
    else:
        module, cls_name = name.rsplit(".", 1)
        cls = getattr(importlib.import_module(module), cls_name)
    assert issubclass(cls, base), f"{cls} is not a subclass of {base}"
    return cls


def create_instance(
    base: t.Type[_T],
    config: t.Dict[str, t.Any],
    *args: t.Any,
    **default_kwargs: t.Any,
) -> _T:
    config = config.copy()
    cls = get_class(base, config.pop("_name"))
    kwargs = {**default_kwargs, **config}
    return cls(*args, **kwargs)
