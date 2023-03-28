import re


class CharsetAdapter:
    """Transforms labels according to the target charset."""

    def __init__(self, target_charset) -> None:
        super().__init__()
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
        self.unsupported = f'[^{re.escape(target_charset)}]'

    def __call__(self, label) -> str:
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        # Remove unsupported characters
        label = re.sub(self.unsupported, '', label)
        return label
