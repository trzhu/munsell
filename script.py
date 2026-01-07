import pandas as pd
import numpy as np
import re, urllib.request
import json
import math
from colour import CCS_ILLUMINANTS, xyY_to_XYZ, XYZ_to_sRGB, XYZ_to_Lab, Lab_to_XYZ, XYZ_to_xyY
from itertools import pairwise
from collections import defaultdict
from scipy.interpolate import LinearNDInterpolator

# 3 was chosen arbitrarily
Y_SCALE = 3
# munsell used illuminant C for his work
ILLUM_C = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["C"]

def load_munsell_dat(url):
    rows = []

    response = urllib.request.urlopen(url)
    lines = response.read().decode('utf-8').splitlines()

    for line in lines[1:]:  # Skip header line
        columns = re.split(r'\s+', line.strip())
        if len(columns) != 6:
            continue
        hue, v, c, x, y, Y = columns
        rows.append({
            "Hue": hue,
            "Value": float(v),
            "Chroma": float(c),
            "x": float(x),
            "y": float(y),
            "Y_lum": float(Y)
        })

    return pd.DataFrame(rows)

# converts from munsell principal/adjacent hue system (RYGBP) to degrees
def munsell_hue_to_deg(hue_str):
    hue_match = re.match(r'(\d{1,2}(\.\d+)?)([A-Z]+)', hue_str.strip())
    if not hue_match:
        return None
    number = float(hue_match.group(1))
    letter = hue_match.group(3)

    hue_order = ['R', 'YR', 'Y', 'GY', 'G', 'BG', 'B', 'PB', 'P', 'RP']
    base_index = hue_order.index(letter)

    return 3.6 * (base_index * 10 + number) # range: 0–360

# Lab to XYZ to RGB
def Lab_to_sRGB(lab):
    xyz = Lab_to_XYZ(lab, illuminant=ILLUM_C)
    sRGB = XYZ_to_sRGB(xyz, illuminant=ILLUM_C)
    sRGB_clipped = np.clip(sRGB, 0, 1)
    is_clipped = np.any(sRGB != sRGB_clipped, axis = 1)
    
    # Handle both single point and array cases
    if is_clipped.shape == ():  # Single point
        return sRGB_clipped, is_clipped.item()
    else:
        return sRGB_clipped, is_clipped

def hvc_to_xyz(h, v, c):
    hue_radians = math.radians(h)
    x = c * math.cos(hue_radians)
    y = Y_SCALE * v
    z = c * math.sin(hue_radians)
    return x, y, z

def xyz_to_hvc(x, y, z):
    h = math.degrees(math.atan2(z, x))
    v = y / Y_SCALE
    c = math.sqrt(x * x + z * z)

    if h < 0:
        h += 360
    return h, v, c

# adds Lab and RGB conversions to the dataframe
# also adds grayscale points to each value plate by avging luminosity
# and white and black
def process(df):
    df["HueDeg"] = df["Hue"].apply(munsell_hue_to_deg)
        
    # munsell's data was recorded as xyY_lum, but Y_lum is scaled from 0-100 instead of 0-1
    xyY = df[["x", "y", "Y_lum"]].to_numpy()
    xyY[:, 2] /= 100  
    
    # Convert xyY to XYZ
    xyz = xyY_to_XYZ(xyY)

    sRGB = XYZ_to_sRGB(xyz, illuminant=ILLUM_C)
    sRGB_clipped = np.clip(sRGB, 0, 1)
    # true if any rgb channel was clipped
    is_clipped = np.any(sRGB != sRGB_clipped, axis=1)

    # Convert to Lab
    Lab = XYZ_to_Lab(xyz, illuminant=ILLUM_C)

    df["X"] = xyz[:, 0]
    df["Y"] = xyz[:, 1]
    df["Z"] = xyz[:, 2]

    df["R"] = sRGB_clipped[:, 0]
    df["G"] = sRGB_clipped[:, 1]
    df["B"] = sRGB_clipped[:, 2]

    df["L*"] = Lab[:, 0]
    df["a*"] = Lab[:, 1]
    df["b*"] = Lab[:, 2]
    
    df["is_clipped"] = is_clipped
    
    grayscale_points = []

    # Grayscale for each Value plate
    # computed by averaging luminosity in CIELAB
    for value, slice_df in df.groupby("Value"):
        avg_L = slice_df["L*"].mean()
        gray_lab = np.array([[avg_L, 0, 0]])

        gray_xyz = Lab_to_XYZ(gray_lab, illuminant=ILLUM_C)
        gray_rgb = np.clip(XYZ_to_sRGB(gray_xyz, illuminant=ILLUM_C), 0, 1)
        gray_xyY = XYZ_to_xyY(gray_xyz)

        grayscale_points.append({
            "Hue": "N",
            "Value": value,
            "Chroma": 0.0,
            "x": gray_xyY[0, 0], "y": gray_xyY[0, 1], "Y_lum": gray_xyY[0, 2],
            "X": gray_xyz[0, 0], "Y": gray_xyz[0, 1], "Z": gray_xyz[0, 2],
            "R": gray_rgb[0, 0], "G": gray_rgb[0, 1], "B": gray_rgb[0, 2],
            "L*": avg_L, "a*": 0.0, "b*": 0.0,
            "HueDeg": 0,
            "is_clipped": False
        })

    # Add black with Value = 0 and white with Value = 10
    for value, L in [(0.0, 0.0), (10.0, 100.0)]:
        lab = np.array([[L, 0, 0]])

        xyz = Lab_to_XYZ(lab, illuminant=ILLUM_C)
        rgb = np.clip(XYZ_to_sRGB(xyz, illuminant=ILLUM_C), 0, 1)
        xyY = XYZ_to_xyY(xyz)

        grayscale_points.append({
            "Hue": "N",
            "Value": value,
            "Chroma": 0.0,
            "x": xyY[0, 0], "y": xyY[0, 1], "Y_lum": xyY[0, 2],
            "X": xyz[0, 0], "Y": xyz[0, 1], "Z": xyz[0, 2], 
            "R": rgb[0, 0], "G": rgb[0, 1], "B": rgb[0, 2],
            "L*": L, "a*": 0.0, "b*": 0.0,
            "HueDeg": 0,
            "is_clipped": False
        })

    df = pd.concat([df, pd.DataFrame(grayscale_points)], ignore_index=True)
    
    return df

# Convert polar Hue/Chroma to X/Z, use Value as Y
def to_3d_coordinates(df):
    radians = np.deg2rad(df["HueDeg"])
    df["X_3D"] = df["Chroma"] * np.cos(radians)
    df["Y_3D"] = df["Value"]
    df["Z_3D"] = df["Chroma"] * np.sin(radians)
    return df

# generate 3d mesh defined by the outermost vertices
# from a df that contains all vertices
# prob gonna depracate this when i clean everything up at the end
def to_mesh_old(df_3d):
    df_exterior = filter_exterior(df_3d)
    return to_mesh(df_exterior)

# creates a mesh from a df that only has exterior vertices
def to_mesh(exterior_df):
    vertices, faces = [], []
    index_map = {}
    global_index = 0
    
    # each slice represents a horizontal "plate" slice
    for value, slice_df in exterior_df.groupby("Value"):
        slice_df = slice_df.sort_values("HueDeg")
        idx_list = []
        
        for _, row in slice_df.iterrows():
            vertices.append((
                row["X_3D"], Y_SCALE * row["Y_3D"], row["Z_3D"],
                row["R"], row["G"], row["B"],
                row["HueDeg"], row["Value"], row["Chroma"],
                row["is_clipped"]
            ))
            idx_list.append(global_index)
            global_index += 1
        
        index_map[value] = idx_list
    
    # get values - vertical axis numbers
    values = sorted(index_map.keys())
    
    # build faces between (vertically) adjacent slices
    for v1, v2 in pairwise(values):
        idx1 = index_map[v1]
        idx2 = index_map[v2]
        
        # bottom cap (black)
        if len(idx1) == 1:
            center = idx1[0]
            for i in range(len(idx2)):
                i_next = (i + 1) % len(idx2)
                faces.append((center, idx2[i], idx2[i_next]))
                
        # top cap (white)
        elif len(idx2) == 1:
            center = idx2[0]
            for i in range(len(idx1)):
                i_next = (i + 1) % len(idx1)
                faces.append((center, idx1[i_next], idx1[i]))
                
        else:
            # regular slice, create quads
            # adaptive quad splitting: split across the shorter diagonal
            N = min(len(idx1), len(idx2))
            for i in range(N):
                i_next = (i + 1) % N
                
                # 3d positions of the 4 quad corners
                v00, v10 = vertices[idx1[i]], vertices[idx1[i_next]]
                v01, v11 = vertices[idx2[i]], vertices[idx2[i_next]]
                
                p00 = np.array([v00[0], v00[1], v00[2]])
                p10 = np.array([v10[0], v10[1], v10[2]])
                p01 = np.array([v01[0], v01[1], v01[2]])
                p11 = np.array([v11[0], v11[1], v11[2]])
                
                #compare diagonal lengths
                diag1 = np.linalg.norm(p11 - p00)
                diag2 = np.linalg.norm(p10 - p01)
                
                # split along shorter diagonal for better shape
                if diag1 < diag2:
                    faces.append((idx1[i], idx2[i], idx2[i_next]))
                    faces.append((idx1[i], idx2[i_next], idx1[i_next]))
                else:
                    faces.append((idx1[i], idx2[i], idx1[i_next]))
                    faces.append((idx2[i], idx2[i_next], idx1[i_next]))
    
    return vertices, faces

# creates a df that contains only the exterior vertices
# (most chromatic vertex at each hue and value)
# and is sorted by value, and within each value, sorted by hue
def filter_exterior(df):
    df_3d = to_3d_coordinates(df)
    
    exterior_rows = []
    
    for value, slice_df in df_3d.groupby("Value"):
        # keep white and black no matter what
        if value in (0.0, 10.0):
            filtered_slice = slice_df
        # otherwise drop grayscales 
        # (prevents issues due to "dummy" 0 HueDeg grayscale vertices)
        else:
            filtered_slice = slice_df[slice_df["Chroma"] > 0]
            
        if not filtered_slice.empty:
            # keep highest chroma per hue
            filtered_slice = (filtered_slice
                .sort_values("Chroma", ascending=False)
                .drop_duplicates("HueDeg", keep="first")
            )
            exterior_rows.append(filtered_slice)
    
    # combine and re-sort
    exterior_df = pd.concat(exterior_rows, ignore_index=True)
    
    return exterior_df

# prepares df for interpolation by adding duplicate grays for each chroma
# and duplicate hue slice at 360 (red)
def interpolate_preprocess(df):
    df_augmented = df.copy()
    df_augmented["is_original"] = True
    df_augmented["flagged_to_drop"] = False
    
    # gonna add some things to help with the interpolation, they all get flagged to be dropped later
    augmented = []
    
    # duplicate grays so that each hue has its own gray
    # helps with chroma interpolation
    grays = df_augmented[df_augmented["Chroma"] == 0]
    
    for _, gray in grays.iterrows():
        for hue in df_augmented["HueDeg"].unique():
            g = gray.copy()
            g["HueDeg"] = hue
            g["flagged_to_drop"] = True
            augmented.append(g)
            
    # duplicate the hue slice at 360 (red), flag it for deletion as well
    hue_360_slice = df_augmented[df_augmented["HueDeg"] == 360].copy()
    hue_360_slice["HueDeg"] = 0
    hue_360_slice["flagged_to_drop"] = True
    
    df_augmented = pd.concat([df_augmented, pd.DataFrame(augmented), hue_360_slice], ignore_index=True)
    
    return df_augmented

# interpolate using LinearNDInterpolator. 
# SNAPS ALL POINTS TO H,V,C GRID. use for point clouds not mesh
def grid_interpolate(df, hue_steps = 2, value_steps=3, chroma_steps=2):
    """
    hue_steps : int
        Number of subdivisions between adjacent hue samples (of the same value and chroma)
    value_steps : int
        Number of subdivisions between adjacent Value layers
    chroma_steps : int
        Number of subdivisions between adjacent Chroma shells (of the same hue and value)
    original data set has: 
        40 hue steps
        11 value steps (inclusive of white and black)
        38 maximum chroma
    """
    df_augmented = interpolate_preprocess(df)
    
    points = df_augmented[["HueDeg", "Value", "Chroma"]].to_numpy()
    
    # create interpolators for Lab values
    interp_L = LinearNDInterpolator(points, df_augmented["L*"])
    interp_a = LinearNDInterpolator(points, df_augmented["a*"])
    interp_b = LinearNDInterpolator(points, df_augmented["b*"])
    
    # build max chroma dictionary
    max_chroma = defaultdict(int)
    for (hue, value), group in df_augmented.groupby(["HueDeg", "Value"]):
        max_chroma[(hue, value)] = group["Chroma"].max()
        
    # write_json(max_chroma)
        
    # original hue/value/chroma spacing is 9, 1, 2 
    hue_stepsize, value_stepsize, chroma_stepsize = 9/hue_steps, 1/value_steps, 2/chroma_steps
    
    hue_grid = np.arange(df["HueDeg"].min(), df["HueDeg"].max(), hue_stepsize)
    value_grid = np.arange(df["Value"].min(), df["Value"].max(), value_stepsize)
    
    for existing_value in sorted(df_augmented["Value"].unique()):
        existing_hues = sorted([hue for (hue, val) in max_chroma.keys() if val == existing_value])
        
        for h1, h2 in pairwise(existing_hues):
            if h2 - h1 != 9.0:
                continue
            
            c1, c2 = max_chroma[(h1, existing_value)], max_chroma[(h2, existing_value)]
            
            # need to do h2+hue_stepsize because it needs to wrap around??
            for h in np.arange(h1, h2 + hue_stepsize, hue_stepsize):
                t = (h - h1) / (h2 - h1)
                max_chroma[(h, existing_value)] = (1-t) * c1 + t * c2
    
    for h in hue_grid:
        existing_vals = sorted([val for (h_key, val) in max_chroma.keys() if h_key == h])
        
        for v1, v2 in pairwise(existing_vals):
            c1, c2 = max_chroma[(h, v1)], max_chroma[(h, v2)]
            for v in value_grid:
                # have to use the whole value grid and limit to between the bounds
                # OR WE MIGHT GET FLOATING POINT MISMATCH LATER
                if v1 < v < v2:
                    t = (v - v1) / (v2 - v1)
                    max_chroma[(h, v)] = (1-t) * c1 + t * c2
    
    new_points = []
    
    for h in hue_grid:
        for v in value_grid:
            if (h, v) in max_chroma:
                max_chroma_limit = max_chroma[(h, v)]
                for c in np.arange(df_augmented["Chroma"].min(), df_augmented["Chroma"].max(), chroma_stepsize):
                    if c > max_chroma_limit:
                        break
                    else:
                        new_points.append([h, v, c])
    
    new_points = np.array(new_points)
    
    # Interpolate Lab values
    new_L = interp_L(new_points)
    new_a = interp_a(new_points)
    new_b = interp_b(new_points)
    
    # filter out any NaN results
    valid_mask = ~(np.isnan(new_L) | np.isnan(new_a) | np.isnan(new_b))
    
    # vectorized Lab_to_sRGB conversion
    valid_points = new_points[valid_mask]
    valid_L = new_L[valid_mask]
    valid_a = new_a[valid_mask]
    valid_b = new_b[valid_mask]

    Lab_array = np.column_stack([valid_L, valid_a, valid_b])
    sRGB_array, is_clipped_array = Lab_to_sRGB(Lab_array)
    
    # build interpolated dataframe
    interpolated_points = pd.DataFrame({
        "HueDeg": valid_points[:, 0],
        "Value": valid_points[:, 1], 
        "Chroma": valid_points[:, 2],
        "L*": valid_L,
        "a*": valid_a,
        "b*": valid_b,
        "R": sRGB_array[:, 0],
        "G": sRGB_array[:, 1],
        "B": sRGB_array[:, 2],
        "is_original": False,
        "is_clipped": is_clipped_array,
        "flagged_to_drop": False
    })
    
    df_result = pd.concat([df, interpolated_points], ignore_index=True)
    
    return df_result

# put all vertices in a point cloud
def to_pointcloud(df_3d):
    vertices = []
    for _, row in df_3d.iterrows():
        x, y, z = row["X_3D"], Y_SCALE * row["Y_3D"], row["Z_3D"]
        r, g, b = row["R"], row["G"], row["B"]
        h, v, c, is_clipped = row["HueDeg"], row["Value"], row["Chroma"], row["is_clipped"]
        vertices.append((x, y, z, r, g, b, h, v, c, is_clipped))
    return vertices

# interpolate along shell surface only. new vertices go on the shell
def shell_interpolate(df, hue_steps = 2, value_steps = 3):
    
    # preprocess
    df_augmented = interpolate_preprocess(df)
    
    # create Lab interpolators for color
    points_hvc = df_augmented[["HueDeg", "Value", "Chroma"]].to_numpy()
    interp_L = LinearNDInterpolator(points_hvc, df_augmented["L*"])
    interp_a = LinearNDInterpolator(points_hvc, df_augmented["a*"])
    interp_b = LinearNDInterpolator(points_hvc, df_augmented["b*"])
    
    # group by (hue, value) and find max chroma at each
    max_chroma_points = []
    for (h, v), group in df_augmented.groupby(["HueDeg", "Value"]):
        max_c = group["Chroma"].max()
        max_chroma_points.append([h, v, max_c])
    
    # build max_chroma interpolator from original data
    # interpolates linearly on triangle faces
    max_chroma_points = np.array(max_chroma_points)
    max_chroma_interp = LinearNDInterpolator(
        # input (h, v)
        max_chroma_points[:, :2],
        # output max chroma
        max_chroma_points[:, 2]
    )
    
    # helper which takes every point in df_exterior (excepte white and black)
    # and takes the weighted average of it with its radial neighbours
    # weight = how much to weight original point
    def radial_smooth(df, weight=0.7):
        df_exterior = df.copy()
        df_exterior = filter_exterior(df)
        
        smoothed_points = []
        
        # work on each "ring" of values
        for v in sorted(df_exterior['Value'].unique()):
            if v == 0 or v == 10:
                continue
            
            ring = df_exterior[df_exterior['Value'] == v].sort_values('HueDeg')
            
            chromas = ring['Chroma'].values
            hues = ring['HueDeg'].values
            n = len(chromas)
            
            for i in range(n):
                # left and right neighbours
                left = chromas[(i - 1) % n]
                right = chromas[(i + 1) % n]
                
                # Weighted average
                nbr_avg = (left + right) / 2
                new_chroma = weight * chromas[i] + (1 - weight) * nbr_avg
                
                smoothed_points.append({
                    'HueDeg': hues[i],
                    'Value': v,
                    'Chroma': new_chroma
                })
        
        # Convert smoothed points to dataframe
        df_smoothed_pts = pd.DataFrame(smoothed_points)
        
        # Compute 3D coordinates
        df_smoothed_pts = to_3d_coordinates(df_smoothed_pts)
        
        # Recompute colors at new (h, v, c) positions
        points_hvc = df_smoothed_pts[["HueDeg", "Value", "Chroma"]].to_numpy()
        new_L = interp_L(points_hvc)
        new_a = interp_a(points_hvc)
        new_b = interp_b(points_hvc)
        
        # Filter valid Lab values
        valid_mask = ~(np.isnan(new_L) | np.isnan(new_a) | np.isnan(new_b))
        
        # Convert Lab to RGB for valid points
        Lab = np.column_stack([new_L[valid_mask], new_a[valid_mask], new_b[valid_mask]])
        sRGB, is_clipped = Lab_to_sRGB(Lab)
        
        # Build complete dataframe with smoothed points
        valid_points = points_hvc[valid_mask]
        valid_xyz = df_smoothed_pts.loc[valid_mask, ['X_3D', 'Y_3D', 'Z_3D']].to_numpy()
        
        df_new_points = pd.DataFrame({
            "HueDeg": valid_points[:, 0],
            "Value": valid_points[:, 1],
            "Chroma": valid_points[:, 2],
            "L*": new_L[valid_mask],
            "a*": new_a[valid_mask],
            "b*": new_b[valid_mask],
            "R": sRGB[:, 0],
            "G": sRGB[:, 1],
            "B": sRGB[:, 2],
            "X_3D": valid_xyz[:, 0],
            "Y_3D": valid_xyz[:, 1],
            "Z_3D": valid_xyz[:, 2],
            "is_original": False,
            "is_clipped": is_clipped,
            "flagged_to_drop": False
        })
        
        # merge the new points back into df
        df = pd.concat([df, df_new_points], ignore_index=True)
        
        return df
    
    df_smoothed_input = radial_smooth(df_augmented)
    
    # rebuild max_chroma interpolator from smoothed data
    smoothed_max_chroma_points = []
    df_smoothed_exterior = filter_exterior(df_smoothed_input)
    for (h, v), group in df_smoothed_exterior.groupby(["HueDeg", "Value"]):
        max_c = group["Chroma"].max()
        smoothed_max_chroma_points.append([h, v, max_c])
    
    smoothed_max_chroma_points = np.array(smoothed_max_chroma_points)
    max_chroma_interp = LinearNDInterpolator(
        smoothed_max_chroma_points[:, :2],
        smoothed_max_chroma_points[:, 2]
    )
    
    # original hue/value/chroma spacing is 9, 1, 2
    hue_stepsize, value_stepsize = 9/hue_steps, 1/value_steps
    
    # sample (hue, value) space finely
    # [start, end) so have to add another step to include value=10
    hue_samples = np.arange(0, 360, hue_stepsize)
    value_samples = np.arange(0, 10 + value_stepsize, value_stepsize)

    query_hv = []
    for v in value_samples:
        for h in hue_samples:
            query_hv.append([h, v])

    query_hv = np.array(query_hv)

    # Get max chroma at all query points
    query_chromas = np.array([max_chroma_interp(h, v) for h, v in query_hv])

    # Filter out invalid chromas
    valid_chroma_mask = ~(np.isnan(query_chromas) | (query_chromas <= 0))
    valid_hv = query_hv[valid_chroma_mask]
    valid_chromas = query_chromas[valid_chroma_mask]

    # Build full (h, v, c) query points for Lab interpolation
    query_hvc = np.column_stack([valid_hv, valid_chromas])

    # Interpolate Lab values
    new_L = interp_L(query_hvc)
    new_a = interp_a(query_hvc)
    new_b = interp_b(query_hvc)

    # Filter out any NaN Lab results
    valid_lab_mask = ~(np.isnan(new_L) | np.isnan(new_a) | np.isnan(new_b))

    # Apply mask to get final valid points
    valid_points = query_hvc[valid_lab_mask]
    valid_L = new_L[valid_lab_mask]
    valid_a = new_a[valid_lab_mask]
    valid_b = new_b[valid_lab_mask]

    # Vectorized Lab_to_sRGB conversion
    Lab_array = np.column_stack([valid_L, valid_a, valid_b])
    sRGB_array, is_clipped_array = Lab_to_sRGB(Lab_array)

    df_result = pd.DataFrame({
        "HueDeg": valid_points[:, 0],
        "Value": valid_points[:, 1],
        "Chroma": valid_points[:, 2],
        "L*": valid_L,
        "a*": valid_a,
        "b*": valid_b,
        "R": sRGB_array[:, 0],
        "G": sRGB_array[:, 1],
        "B": sRGB_array[:, 2],
        "is_original": False,
        "is_clipped": is_clipped_array,
        "flagged_to_drop": False
    })
    
    # delaunay triangulation filters out white and black because chroma = 0
    # add black and white manually
    bw = pd.DataFrame({
        "HueDeg": [0.0, 0.0],
        "Value": [0.0, 10.0],
        "Chroma": [0.0, 0.0],
        "L*": [0.0, 100.0],
        "a*": [0.0, 0.0],
        "b*": [0.0, 0.0],
        "R": [0.0, 1.0],
        "G": [0.0, 1.0],
        "B": [0.0, 1.0],
        "is_original": [True, True],
        "is_clipped": [False, False],
        "flagged_to_drop": [False, False]
    })
    
    df_result = pd.concat([df_result, bw], ignore_index=True)
    
    # write smoothed max chroma to json
    max_chroma = defaultdict(int)
    for (hue, value), group in df_result.groupby(["HueDeg", "Value"]):
        max_chroma[(hue, value)] = group["Chroma"].max()
    
    write_json(max_chroma)

    return df_result


def to_smooth_mesh(df):
    # df = shell_interpolate(df)
    # df = filter_exterior(df)
    # df = to_3d_coordinates(df)
    
    df1 = shell_interpolate(df)
    print("After shell_interpolate:", df1.columns.tolist())
    
    df2 = filter_exterior(df1)
    print("After filter_exterior:", df2.columns.tolist())
    
    df3 = to_3d_coordinates(df2)
    print("After to_3d_coordinates:", df3.columns.tolist())
    
    return to_mesh(df3)
    
    # return value: vertices, faces
    return to_mesh(df)

def write_ply(vertices, faces, filename):
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float hue\n")
        f.write("property float value\n")
        f.write("property float chroma\n")
        f.write("property uchar is_clipped\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for x, y, z, r, g, b, h, v, c, is_clipped in vertices:
            r_byte = int(r * 255)
            g_byte = int(g * 255)
            b_byte = int(b * 255)
            is_clipped_byte = int(str(is_clipped).strip().lower() == "true")
            f.write(f"{x} {y} {z} {r_byte} {g_byte} {b_byte} {h} {v} {c} {is_clipped_byte}\n")

        for face in faces:
            f.write(f"3 {' '.join(map(str, face))}\n")


# write a dictionary to json to read later
def write_json(dictionary):
    # convert from default dict to regular dict
    d = dict(dictionary)

    nested_dict = {}
    for (hue, value), chroma in d.items():
        if hue not in nested_dict:
            nested_dict[hue] = {}
        nested_dict[hue][value] = chroma
    
    with open('max_chroma.json', 'w') as f:
        json.dump(nested_dict, f, indent=2)
    
    print("wrote to max_chroma.json")

# point cloud of only original dataset
def to_pointcloud_original():
    df_processed = pd.read_csv("munsell_parsed.csv", index_col=False)
    df_3d = to_3d_coordinates(df_processed)
    
    vertices = to_pointcloud(df_3d)
    write_ply(vertices, [], "munsell_pointcloud_original.ply")


def main():
    df_processed = pd.read_csv("munsell_parsed.csv", index_col=False)
    vertices, faces = to_smooth_mesh(df_processed)
    write_ply(vertices, faces, "munsell_mesh.ply")
    
    print(":)")


if __name__ == "__main__":
    main()
