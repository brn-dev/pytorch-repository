import abc

from src.utils import remove_duplicates_keep_order

Tags = list[str]

class HasTags(abc.ABC):

    # noinspection PyMethodMayBeStatic
    def collect_tags(self) -> Tags:
        return []

    @staticmethod
    def combine_tags(*tag_sources: Tags) -> Tags:
        return remove_duplicates_keep_order([
            tag
            for tags in tag_sources
            for tag in tags
        ])
