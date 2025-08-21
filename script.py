import pandas as pd
import numpy as np
import re, urllib.request
from colour import CCS_ILLUMINANTS, xyY_to_XYZ, XYZ_to_sRGB, XYZ_to_Lab, Lab_to_XYZ, XYZ_to_xyY
from itertools import pairwise
from collections import defaultdict
from scipy.interpolate import LinearNDInterpolator

# TODO compute a scaling factor that makes the Y axis perceptually uniform?
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
    else:  # Array of points
        return sRGB_clipped, is_clipped
    
    # because is_clipped is an np array
    # return sRGB_clipped, is_clipped.item()

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
# TODO: isnt great with fully interpolated mesh, not sure why rn
def to_mesh(df_3d):
    # each entry of the dictionary is a df of all points with that Value
    # represents a horizontal "plate", which are stacked to form the space
    slices = {}
    for v, slice_df in df_3d.groupby("Value"):
        
        # remove grayscale (except black and white)
        if v not in (0.0, 10.0):
            slice_df = slice_df[slice_df["Chroma"] > 0]

        # skip slices that were interpolated to only have a singular grayscale point
        if slice_df.empty:
            continue
        
        # sort each slice by hue
        # # only keep the highest chroma (outermost) vertex of each slice
        slice_df = slice_df.sort_values("Chroma", ascending=False).drop_duplicates("HueDeg", keep="first")
        slice_df = slice_df.sort_values("HueDeg")
        slices[v] = slice_df
    
    # originally 1-9 for munsell data, 0-10 including white and black
    values = sorted(slices.keys())
    
    vertices, faces = [], []
    index_map = {}
    global_index = 0
    
    # add all vertices to global vertices list
    for v in values:
        slice_df = slices[v]
        idx_list = []
        for _, row in slice_df.iterrows():
            vertices.append((row["X_3D"], Y_SCALE * row["Y_3D"], row["Z_3D"], 
                             row["R"], row["G"], row["B"], 
                             row["HueDeg"], row["Value"], row["Chroma"],
                             row["is_clipped"]))
            idx_list.append(global_index)
            global_index += 1
        # and index them (they will automatically go in by hue order)
        index_map[v] = idx_list

    # build faces between adjacent slices
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
            N = min(len(idx1), len(idx2))
            for i in range(N):
                i_next = (i + 1) % N
                # form two triangles (that make 1 quad)
                faces.append((idx1[i], idx2[i], idx2[i_next]))
                faces.append((idx1[i], idx2[i_next], idx1[i_next]))
     
    return vertices, faces

# new interpolate using LinearNDInterpolator
# TODO: still some bugs where some layers are missing
# and handling of caps is awful
def interpolate(df, df_all = None, hue_steps = 2, value_steps=3, chroma_steps=2):
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
    target: 
        80+ hue steps
        20-30+ value steps
        30+ chroma steps
    """
    
    df = df.copy()
    df["is_original"] = True
    df["flagged_to_drop"] = False
    
    # gonna add some things to help with the interpolation, they all get flagged to be dropped later
    augmented = []
    
    # duplicate grays so that each hue has its own gray
    # helps with chroma interpolation
    grays = df[df["Chroma"] == 0]
    
    for _, gray in grays.iterrows():
        for hue in df["HueDeg"].unique():
            g = gray.copy()
            g["HueDeg"] = hue
            g["flagged_to_drop"] = True
            augmented.append(g)
            
    # duplicate the hue slice at 360 (red), flag it for deletion as well
    hue_360_slice = df[df["HueDeg"] == 360].copy()
    hue_360_slice["HueDeg"] = 0
    hue_360_slice["flagged_to_drop"] = True
    
    df_augmented = pd.concat([df, pd.DataFrame(augmented), hue_360_slice], ignore_index=True)
    
    points = df_augmented[["HueDeg", "Value", "Chroma"]].to_numpy()
    
    # create interpolators for Lab values
    interp_L = LinearNDInterpolator(points, df_augmented["L*"])
    interp_a = LinearNDInterpolator(points, df_augmented["a*"])
    interp_b = LinearNDInterpolator(points, df_augmented["b*"])
    
    # build max chroma dictionary
    max_chroma = defaultdict(int)
    for (hue, value), group in df.groupby(["HueDeg", "Value"]):
        max_chroma[(hue, value)] = group["Chroma"].max()
    
    # original hue/value/chroma spacing is 9, 1, 2 
    hue_stepsize, value_stepsize, chroma_stepsize = 9/hue_steps, 1/value_steps, 2/chroma_steps
    
    for existing_value in sorted(df_augmented["Value"].unique()):
        existing_hues = sorted([hue for (hue, val) in max_chroma.keys() if val == existing_value])
        
        for h1, h2 in pairwise(existing_hues):
            c1, c2 = max_chroma[(h1, existing_value)], max_chroma[(h2, existing_value)]
            
            # need to do h2+hue_stepsize because it needs to wrap around??
            # idk it's not working perfectly
            # interpolate between h1 and h2
            for h in np.arange(h1, h2 + hue_stepsize, hue_stepsize):
                t = (h - h1) / (h2 - h1)
                max_chroma[(h, existing_value)] = (1-t) * c1 + t * c2
    
    for h in np.arange(df_augmented["HueDeg"].min(), df_augmented["HueDeg"].max(), hue_stepsize):
        # TODO not sure if this is right
        all_vals = sorted([val for (h_key, val) in max_chroma.keys() if h_key == h])
        
        for v1, v2 in pairwise(all_vals):
            c1, c2 = max_chroma[(h, v1)], max_chroma[(h, v2)]
            
            for v in np.arange(v1, v2, value_stepsize):
                t = (v - v1) / (v2 - v1)
                max_chroma[(h, v)] = (1-t) * c1 + t * c2
    
    new_points = []
    
    for h in np.arange(df_augmented["HueDeg"].min(), df_augmented["HueDeg"].max(), hue_stepsize):
        for v in np.arange(df_augmented["Value"].min(), df_augmented["Value"].max(), value_stepsize):
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
    
    # df_cleaned = df[~df["flagged_to_drop"]].drop(columns=["flagged_to_drop"]).reset_index(drop=True)
    
    df_result = pd.concat([df, interpolated_points], ignore_index=True)
    
    return df_result

# Interpolates the Munsell renotation dataset along each of Hue, Value, Chroma
# perform the interpolation in CIELAB.
# original data set has: 
#   40 hue steps
#   11 value steps (inclusive of white and black)
#   38 maximum chroma
# target: 
#   80+ hue steps
#   20-30+ value steps
#   30+ chroma steps
# def interpolate(df, df_all, hue_steps=2, value_steps=2, chroma_steps=3):
    """
    hue_steps : int
        Number of subdivisions between adjacent hue samples (of the same value and chroma)
    value_steps : int
        Number of subdivisions between adjacent Value layers
    chroma_steps : int
        Number of subdivisions between adjacent Chroma shells (of the same hue and value)

    Returns
    DataFrame
        Combined dataframe with original and interpolated points, 
        with `is_original` flag.
    """
    df = df.copy()
    df["is_original"] = True
    
    df, max_chroma = interpolate_chroma(df, chroma_steps)
    
    df = interpolate_value(df, value_steps)
    
    df = interpolate_hue(df, hue_steps)
    
    df = interpolate_edges(df, df_all, max_chroma)
    
    return df

# interpolate radially, in cylindrical shells
# returns df + max chroma at each point
def interpolate_chroma(df, chroma_steps = 3):
    # this section duplicates grays so that each hue slice can have its own gray
    # helps with radial interpolation
    grays = df[df["Chroma"] == 0]
    augmented = []

    for _, gray in grays.iterrows():
        for hue in df["HueDeg"].unique():
            g = gray.copy()
            g["HueDeg"] = hue
            augmented.append(g)
    df = pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)
    
    all_points = []
    
    # build a map of maximum chroma at each (value, hue) position
    # this is so when I interpolate later, I can reference the max chroma of neighbouring "spokes"
    max_chroma = {}
    
    # interpolate radially along Chroma axis
    # splits the df into buckets that have the same value and huedeg
    # so we can interpolate between chroma
    for (value, hue), group in df.groupby(["Value", "HueDeg"]):
        # save for later
        max_chroma[(value, hue)] = group["Chroma"].max()
        
        group = group.sort_values("Chroma")
        chroma_pts = group[["Chroma", "L*", "a*", "b*"]].to_numpy()
        for i in range(len(chroma_pts) - 1):
            p1 = chroma_pts[i]
            p2 = chroma_pts[i+1]
            for t in np.linspace(0, 1, chroma_steps+2)[1:-1]:
                # interpolate in Lab, then convert lab to rgb
                lerp = (1-t)*p1 + t*p2
                chroma, L, a, b = lerp
                
                Lab = np.array([[L, a, b]])
                sRGB, is_clipped = Lab_to_sRGB(Lab)
                
                all_points.append({
                    "HueDeg": hue,
                    "Value": value,
                    "Chroma": chroma,
                    "L*": L, "a*": a, "b*": b,
                    "R": sRGB[0, 0], "G": sRGB[0, 1], "B": sRGB[0, 2],
                    "is_original": False,
                    "is_clipped": is_clipped
                })
    
    # drop the extra grayscales now that we're done with them
    df = df[~((df["Chroma"] == 0) & (df["HueDeg"] != 0))].reset_index(drop=True)
    
    df = pd.concat([df, pd.DataFrame(all_points)], ignore_index=True)
    return df, max_chroma

# interpolate vertically, between Value "plates"
def interpolate_value(df, value_steps):
    all_points = [] 
    for (chroma, hue), group in df.groupby(["Chroma", "HueDeg"]):
        group = group.sort_values("Value")
        value_pts = group[["Value", "L*", "a*", "b*"]].to_numpy()
        for i in range(len(value_pts) - 1):
            p1 = value_pts[i]
            p2 = value_pts[i+1]
            for t in np.linspace(0, 1, value_steps+2)[1:-1]:
                lerp = (1-t)*p1 + t*p2
                value, L, a, b = lerp
                
                Lab = np.array([[L, a, b]])
                sRGB, is_clipped = Lab_to_sRGB(Lab)
                
                all_points.append({
                    "HueDeg": hue,
                    "Value": value,
                    "Chroma": chroma,
                    "L*": L, "a*": a, "b*": b,
                    "R": sRGB[0, 0], "G": sRGB[0, 1], "B": sRGB[0, 2],
                    "is_original": False,
                    "is_clipped": is_clipped
                })

    df = pd.concat([df, pd.DataFrame(all_points)], ignore_index=True)    
    return df


# interpolate circumferentially, between hues, within each Value "plate"
def interpolate_hue(df, hue_steps):
    all_points = []
    
    for (value, chroma), group in df.groupby(["Value", "Chroma"]):
                
        group = group.sort_values("HueDeg")
        hue_pts = group[["HueDeg", "L*", "a*", "b*"]].to_numpy()
        
        for i in range(len(hue_pts)):
            p1 = hue_pts[i]
            p2 = hue_pts[(i+1) % len(hue_pts)]
            
            # if angular difference between two spokes isn't 9.0, they aren't actually next to each other
            hue_delta = (p2[0] - p1[0]) % 360
            if hue_delta != 9.0:
                continue
            
            for t in np.linspace(0, 1, hue_steps+2)[1:-1]:
                lab_lerp = (1-t)*p1[1:] + t*p2[1:]
                L, a, b = lab_lerp
                
                # cant lerp HueDeg - handle the seam from 360 to 0
                if i == len(hue_pts) - 1 or p2[0] < p1[0]:
                    hue = ((1-t)*p1[0] + t*(p2[0]+360)) % 360
                else:
                    hue = ((1-t)*p1[0] + t*p2[0]) % 360
                
                Lab = np.array([[L, a, b]])
                sRGB, is_clipped = Lab_to_sRGB(Lab)
                
                all_points.append({
                    "HueDeg": hue,
                    "Value": value,
                    "Chroma": chroma,
                    "L*": L, "a*": a, "b*": b,
                    "R": sRGB[0, 0], "G": sRGB[0, 1], "B": sRGB[0, 2],
                    "is_original": False,
                    "is_clipped": is_clipped,
                })
    
    df = pd.concat([df, pd.DataFrame(all_points)], ignore_index=True)
    
    df["HueDeg"] = np.round(df["HueDeg"], 6)  # Fix floating point precision
    
    return df

def interpolate_edges(df, df_all, max_chroma, chroma_steps = 3):
    # after simple interpolation, detect which new spokes need extension based on their neighbours
    extension_points = []
    
    # group by (Value, HueDeg) to work spoke by spoke
    # one dimension at a time
    # first, find hue neighbours (left and right neighbours)
    for (value, hue), spoke in df[df['Value'] % 1 == 0].groupby(["Value", "HueDeg"]):
        # for floating point issues
        value, hue = np.round(value), np.round(hue)
        
        # skip black and white and grayscale
        if value in [0, 10] or hue == 0:
            continue
        
        # skip original spokes
        if spoke["is_original"].all():
            continue
        
        current_max_chroma = spoke["Chroma"].max()
        
        # in the original data, HueDeg has intervals of 9
        left_hue = 9 * (hue // 9)
        right_hue = (9 + left_hue) % 360
        # t for hue lerp
        t = (hue % 9) / 9
        
        # Calculate what this spoke's length should be
        # by lerping between left and right max chromas
        target_max_chroma = (1-t) * max_chroma[(value, left_hue)] + t * max_chroma[(value, right_hue)]
        
        # np.arange is range with floats
        for chroma in np.arange(current_max_chroma, target_max_chroma, 1/chroma_steps):
            # placeholder: place gray points (it works)
            extension_points.append({
                "HueDeg": hue,
                "Value": value,
                "Chroma": chroma,
                "L*": 1, "a*": 0, "b*": 0,
                "R": 0.5, "G": 0.5, "B": 0.5,
                "is_original": False,
                "is_clipped": False,
            })
    
    # NEXT, find value neighbours (above & below)
    # in the original data, value is in intervals of one
    for (value, hue), spoke in df[df["HueDeg"] % 9 == 0].groupby(["Value", "HueDeg"]):
        
        # skip black and white and grayscale
        if value in [0, 10] or hue == 0:
            continue
        
        # skip original spokes
        if spoke["is_original"].all():
            continue
        
        current_max_chroma = spoke["Chroma"].max()
        
        below_val = np.floor(value)
        t = value - below_val
        
        target_max_chroma = (1 - t) * max_chroma[(below_val, hue)] + t * max_chroma[(below_val + 1, hue)]
        
        for chroma in np.arange(current_max_chroma, target_max_chroma, 1/chroma_steps):
            # placeholder gray points
            extension_points.append({
                "HueDeg": hue,
                "Value": value,
                "Chroma": chroma,
                "L*": 1, "a*": 0, "b*": 0,
                "R": 0.5, "G": 0.5, "B": 0.5,
                "is_original": False,
                "is_clipped": False,
            })
            
    df = pd.concat([df, pd.DataFrame(extension_points)], ignore_index=True)
    return df

# helper for interpolate
def munsell_pt_to_lab(ref_point):
    xyY = np.array([ref_point['x'], ref_point['y'], ref_point['Y_lum']/100])
    xyz = xyY_to_XYZ(xyY.reshape(1, -1))
    lab = XYZ_to_Lab(xyz, illuminant=ILLUM_C)[0]
    return lab

# put all vertices in a point cloud
def to_pointcloud(df_3d):
    vertices = []
    for _, row in df_3d.iterrows():
        # TODO: add additional vertex metadata for HueDeg, Chroma, and Value
        # and isClipped
        x, y, z = row["X_3D"], Y_SCALE * row["Y_3D"], row["Z_3D"]
        r, g, b = row["R"], row["G"], row["B"]
        h, v, c, is_clipped = row["HueDeg"], row["Value"], row["Chroma"], row["is_clipped"]
        vertices.append((x, y, z, r, g, b, h, v, c, is_clipped))
    return vertices

def write_obj(vertices, faces, filename="munsell_mesh.obj"):
    with open(filename, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # obj format uses 1-based indexing
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

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
            # csv may or may not preserve types, so it could be a bool or a string lol
            # if isinstance(is_clipped, bool):
            #     is_clipped_byte = int(is_clipped)
            # else:
            is_clipped_byte = int(str(is_clipped).strip().lower() == "true")
            f.write(f"{x} {y} {z} {r_byte} {g_byte} {b_byte} {h} {v} {c} {is_clipped_byte}\n")

        for face in faces:
            f.write(f"3 {' '.join(map(str, face))}\n")



def to_pointcloud_original():
    df_processed = pd.read_csv("munsell_parsed.csv", index_col=False)
    df_3d = to_3d_coordinates(df_processed)
    
    vertices = to_pointcloud(df_3d)
    write_ply(vertices, [], "munsell_pointcloud_original.ply")
    
def with_interpolation():
    # input_url = "https://www.rit-mcsl.org/MunsellRenotation/real.dat"
    # df_raw = load_munsell_dat(input_url)
    
    # df_processed = process(df_raw)
    # df_processed.to_csv("munsell_parsed.csv", index=False)
    # print("saved to munsell_parsed.csv")
    
    all_url = "https://www.rit-mcsl.org/MunsellRenotation/all.dat"
    df_all = load_munsell_dat(all_url)
    # df_all.to_csv("munsell_all.csv", index = False)
    
    df_processed = pd.read_csv("munsell_parsed.csv", index_col=False)
    
    df_interpolated = interpolate(df_processed, df_all)
    df_interpolated.to_csv("munsell_interpolated.csv", index=False)
    print("saved to munsell_interpolated.csv")
    
    df_3d = to_3d_coordinates(df_interpolated)
    df_3d.to_csv("munsell_3d.csv", index=False)
    print("saved to munsell_3d.csv")
    
    # df_3d = pd.read_csv("munsell_3d.csv", index_col=False)
    
    # create a point cloud
    vertices = to_pointcloud(df_3d)
    write_ply(vertices, [], "munsell_pointcloud_interpolated.ply")
    
    # create a "shell" mesh
    outer_vertices, faces = to_mesh(df_3d)
    write_ply(outer_vertices, faces, "munsell_mesh.ply")

# point cloud and mesh with only the original dataset
def original():
    # input_url = "https://www.rit-mcsl.org/MunsellRenotation/real.dat"
    # df_raw = load_munsell_dat(input_url)
    
    # df_processed = process(df_raw)
    # df_processed.to_csv("munsell_parsed.csv", index=False)
    # print("saved to munsell_parsed.csv")
    
    df_processed = pd.read_csv("munsell_parsed.csv", index_col=False)
    
    df_3d = to_3d_coordinates(df_processed)
    df_3d.to_csv("munsell_3d.csv", index=False)
    print("saved to munsell_3d.csv")
    
    # create a point cloud
    vertices = to_pointcloud(df_3d)
    write_ply(vertices, [], "munsell_pointcloud.ply")
    
    # create a "shell" mesh
    outer_vertices, faces = to_mesh(df_3d)
    write_ply(outer_vertices, faces, "munsell_mesh.ply")


def main():
    df_processed = pd.read_csv("munsell_parsed.csv", index_col=False)
    
    df_interpolated = interpolate(df_processed)
    df_interpolated.to_csv("munsell_interpolated.csv", index=False)
    print("saved to munsell_interpolated.csv")
    
    df_3d = to_3d_coordinates(df_interpolated)
    df_3d.to_csv("munsell_3d.csv", index=False)
    print("saved to munsell_3d.csv")
    
    # create a point cloud
    vertices = to_pointcloud(df_3d)
    write_ply(vertices, [], "munsell_pointcloud_interpolated.ply")
    
    # create a "shell" mesh
    outer_vertices, faces = to_mesh(df_3d)
    write_ply(outer_vertices, faces, "munsell_mesh.ply")
    
    print(":)")
    


if __name__ == "__main__":
    main()
