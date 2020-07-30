# functionalstream

## Installation

```shell script
pip install functionalstream
```

## Example

### Stream

```python
from functionalstream import Stream

# lst1 = [0, 6, 12, 18, 24]
lst1 = Stream(range(10)).filter(lambda x: x % 2 == 0).map(lambda x: x * 3).to_list()

# lst2 = [(1, 2), (3, 4)]
lst2 = Stream([(1,2), (3,4), (6,5)]).filter(lambda x, y: x < y, star=True).to_list()

# array = array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
array = Stream(range(10)).to_numpy_array(dtype=float)

# for function which has side effect
Stream(range(10)).apply(lambda x: print(x))
```

### PathStream

```python
from functionalstream import PathStream

video_paths = PathStream.from_path('./').filter_extension(extensions={".jpeg"})
```


### Pipeline

```python
from functionalstream.functions import increment
from functionalstream import Stream

# x = 3
x = Stream([increment] * 3).pipeline(0)
```

