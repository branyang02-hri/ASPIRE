# Canonical Spielberg Assets

`Spielberg_map.png` and `Spielberg_map.yaml` are copied byte-for-byte from
`honda-research-institute/f1tenth_gym_ros` commit
`5da9d22b3a94931522388ecd2df5238e5a0295c9`. The map is the original
2000 x 2000 grayscale occupancy image; ASPIRE does not modify it.

`spielberg_reference_path.csv` is a deterministic 0.20 m centerline generated
from that map's two boundary loops. It contains 1,759 waypoints over 343.96 m.

SHA-256:

```text
3378ea1da231e00b4c41015311f7f3a63fb62420d8c54aaa23b6d1ca7c947e56  Spielberg_map.png
86c0eb7546bb035ee6e0173eddc8a2953f8c079944acd28374856f8cc245f6f1  Spielberg_map.yaml
4c647bac61d3c9ce69ffec26bcd570c07874f071ae40eea1e3d8e61517522c3a  spielberg_reference_path.csv
```
