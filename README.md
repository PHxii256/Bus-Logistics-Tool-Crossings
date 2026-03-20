>[!NOTE]
> You need Docker for this. If you are on Windows, it's advised to use WSL.

### Create a Folder and Download Egypt's map:

```bash
wget http://download.geofabrik.de/africa/egypt-latest.osm.pbf
```

### Extract the map (builds the road network):

# 2. Extract cleanly
```bash
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/egypt-latest.osm.pbf
```
# 3. Contract cleanly
```bash
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-contract /data/egypt-latest.osrm
```
# 4. Start the server again
```bash
docker run -d -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --max-table-size 8000 /data/egypt-latest.osrm
```


Done. The routing engine is now available at `http://localhost:5000`.


### Results

check `780ff238` folder in expirement 3 for inout_caps_off_high.json

check `055b75b5` folder in expirement 3 for input_caps_off.json


#### Drive Crossings Best Result

preview_bb4435e8_1773991950 in experieemnt 4

even better results:

preview_83d80b8f_1773993404

#### want to reproduce? 

`python3 experiments/experiment4_crossings/preview_named_crossings_2.py --debug`
