# Canonical Levine Assets

These files are copied byte-for-byte from `honda-research-institute/f1tenth_gym_ros` commit `5da9d22b3a94931522388ecd2df5238e5a0295c9`:

- `maps/levine.png`
- `maps/levine.yaml`
- `ref_path/levine_reference_path.csv`

SHA-256:

```text
f5983fdc8e1a2395f533502a582089d0c0b76835523f4d21a00982da4af2b29e  levine.png
6c422686da9fdc28ace3613aae31520102225516c2f09aea51b6aedce383c1fe  levine.yaml
b9406f5bf11bad0db668002fa37c772e36ec283cdc595e062cfea85197b6f586  levine_reference_path.csv
```

ASPIRE does not create a derived map asset. The pinned F1TENTH Gym performs its standard in-memory occupancy thresholding when loading the original PNG.
The resulting `2048 x 2048` `float64` map array has SHA-256
`234b5bcc50ea02fd3845897f88bbc6066984f200ffb1ed664734d19015b0a315`.
