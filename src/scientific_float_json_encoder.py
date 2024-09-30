from json import JSONEncoder
from json.encoder import encode_basestring_ascii, _make_iterencode


class ScientificFloatJsonEncoder(JSONEncoder):

    def __init__(
            self, *, decimal_precision: float = 4, skipkeys=False, ensure_ascii=True,
            check_circular=True, allow_nan=True, sort_keys=False,
            indent=None, separators=None, default=None
    ):
        super().__init__(
            skipkeys=skipkeys, ensure_ascii=ensure_ascii,
            check_circular=check_circular, allow_nan=allow_nan, sort_keys=sort_keys,
            indent=indent, separators=separators, default=default
        )
        self.decimal_precision = decimal_precision

    def iterencode(self, o, _one_shot=False):
        # Define a custom float formatting function
        def floatstr(
                o,
                allow_nan=self.allow_nan,
                _repr=float.__repr__,
                _inf=float('inf'),
                _neginf=-float('inf')
        ):
            if o != o:
                # NaN
                text = 'NaN'
            elif o == _inf:
                text = 'Infinity'
            elif o == _neginf:
                text = '-Infinity'
            else:
                # Format the float using '.4e'
                return format(o, '.4e')
            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " + repr(o)
                )
            return text

        # Use the original encoder's settings
        if self.check_circular:
            markers = {}
        else:
            markers = None

        _encoder = encode_basestring_ascii

        # Create an iterator that will encode the object
        _iterencode = _make_iterencode(
            markers,
            self.default,
            _encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            self.allow_nan,
            _one_shot
        )
        return _iterencode(o, 0)
