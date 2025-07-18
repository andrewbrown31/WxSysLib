from geopy.distance import geodesic
import numpy as np

def create_starf_point(statdir, lon, lat, lev_range):
	starf_point = open(f'{statdir}','w')
	s = 0
	for lev in np.array(lev_range):
		starf_point.write('0.00 '+ str(lon).rjust(2,'0') + ' ' + str(lat).rjust(2,'0') + ' '+str(lev))
		s = s + 1

def create_starf_box(statdir, lon_min, lon_max, lat_min, lat_max, interval, lev_range):
    starf_box= open(f'{statdir}','w')
    s = 0
    for lev in np.array(lev_range):
        for lon in np.arange(lon_min, lon_max+interval, interval):
            for lat in np.arange(lat_min,lat_max+interval, interval):
                starf_box.write('0.00 '+ str(lon).rjust(2,'0') + ' ' + str(lat).rjust(2,'0') + ' '+str(lev)+'\n')
                s = s + 1

def create_starf_line(statdir, lon_sta, lon_end, lat_sta, lat_end, interval, lev_range):
    starf_line= open(f'{statdir}','w')
    s = 0
    lons = np.arange(lon_sta, lon_end+interval, interval)
    lats = np.arange(lat_sta, lat_end+interval, interval)
    for lev in np.array(lev_range):
         for lon, lat in zip(lons, lats):
            starf_line.write('0.00 '+ str(lon).rjust(2,'0') + ' ' + str(lat).rjust(2,'0') + ' '+str(lev)+'\n')
            s = s + 1


def create_circ_starf_circle(statdir, clon, clat, max_radius_km, spacing_km, lev_range):
	starf_box = open(f'{statdir}', 'w')
	s = 0
	n_rings = int(max_radius_km//spacing_km)
	for lev in lev_range:
		for i in range(n_rings + 1):
			r = i * spacing_km
			if r == 0:
				starf_box.write(f"0.00 {clon:.2f} {clat:.2f} {lev}\n")
				s += 1
			else:
				n_points = max(6, int(round(2 * np.pi * r / spacing_km)))
				for angle in np.linspace(0, 360, n_points, endpoint=False):
					dest = geodesic(kilometers=r).destination((clat, clon), angle)
					starf_box.write(f"0.00 {dest.longitude:.2f} {dest.latitude:.2f} {lev}\n")
					s += 1
