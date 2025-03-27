from typing import Any, Dict, Optional, Tuple



def get_factory_adder() -> Tuple[Any, Dict[str, Any]]:
    """Get a function that adds a class to a list and the corresponding
    list. Useful for creating a factory with a list of classes. The
    intended use is as a decorator.
    You can also can specify a different name for the class in the list,
    to use it at creation time instead of the class name.
    Example:
        >>> add_class, classes_dict = get_factory_adder()
        >>> @add_class
        ... class A:
        ...     pass
        >>> @add_class('Cc')
        ... class C:
        ...     pass
    """
    classes_dict = {}
    def _add_class(class_: Any, name: Optional[str]=None) -> Any:
        if name is None:
            name = class_.__name__
        classes_dict[name] = class_
        return class_

    def add_class(class_: Any, name: Optional[str]=None) -> Any:
        if not callable(class_):
            name = class_
            def wrapper(class_: Any) -> Any:
                return _add_class(class_, name)
            return wrapper
        else:
            return _add_class(class_)

    return add_class, classes_dict
