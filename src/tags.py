import abc

Tags = list[str]

class HasTags(abc.ABC):

    # noinspection PyMethodMayBeStatic
    def collect_tags(self) -> Tags:
        return []
