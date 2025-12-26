import numpy as np
import pyproj

botleft=(285910.6172933511, 5633312.5510845855)
topright=(292625.88221252145, 5641034.998405547)

N = 20

rng = np.random.default_rng()
points = rng.uniform(botleft, topright, (N, 2))

# print(points)
# convert to lon lat
t = pyproj.Transformer.from_crs('EPSG:32632', 'EPSG:4326', always_xy=True)
points_lonlat = np.stack(t.transform(points[:,0], points[:,1]), axis=-1)
print(points_lonlat)


res = [
    dict(
        id=f'Receiver {i+1}',
        latitude=p[1],
        longitude=p[0]
    )
    for i, p in enumerate(points_lonlat)
]

import json

print(json.dumps(res))
