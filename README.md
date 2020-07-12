# pyiterator

## Installation

```shell script
pip install pyiterator
```

## Example

```python
from pyiterator import Stream

# stream = [0, 6, 12, 18, 24]
stream = Stream(range(10)).filter(lambda x: x % 2 == 0).map(lambda x: x * 3).to_list()
```
