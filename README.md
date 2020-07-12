# functionalstream

## Installation

```shell script
pip install functionalstream
```

## Example

```python
from functionalstream import Stream

# stream = [0, 6, 12, 18, 24]
stream = Stream(range(10)).filter(lambda x: x % 2 == 0).map(lambda x: x * 3).to_list()

# stream = [(1, 2), (3, 4)]
stream = Stream([(1,2), (3,4), (6,5)]).filter(lambda x, y: x < y, star=True).to_list()
```
