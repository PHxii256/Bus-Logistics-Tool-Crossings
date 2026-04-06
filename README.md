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