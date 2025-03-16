class DictAttrAccess:
    """Mixin that enables both dictionary-style and attribute access to fields."""

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def keys(self) -> List[str]:
        return [f.name for f in fields(self)]

    def values(self) -> List[Any]:
        return [getattr(self, f.name) for f in fields(self)]

    def items(self) -> List[tuple]:
        return [(f.name, getattr(self, f.name)) for f in fields(self)]

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**data)