import pandas as pd
import numpy as np
import re, urllib.request
from colour import CCS_ILLUMINANTS, xyY_to_XYZ, XYZ_to_sRGB, XYZ_to_Lab, Lab_to_XYZ, XYZ_to_xyY
from itertools import pairwise
from scipy.interpolate import griddata

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

def process(df):
    df["HueDeg"] = df["Hue"].apply(munsell_hue_to_deg)
        
    # munsell's data was recorded as xyY_lum, but Y_lum is scaled from 0-100 instead of 0-1
    xyY = df[["x", "y", "Y_lum"]].to_numpy()
    xyY[:, 2] /= 100  
    
    # Convert xyY to XYZ
    XYZ = xyY_to_XYZ(xyY)

    sRGB = XYZ_to_sRGB(XYZ, illuminant=ILLUM_C)
    sRGB_clipped = np.clip(sRGB, 0, 1)
    # TODO: IF CLIPPED should I write store that at the vertex or something?

    # Convert to Lab
    Lab = XYZ_to_Lab(XYZ, illuminant=ILLUM_C)

    df["X"] = XYZ[:, 0]
    df["Y"] = XYZ[:, 1]
    df["Z"] = XYZ[:, 2]

    df["R"] = sRGB_clipped[:, 0]
    df["G"] = sRGB_clipped[:, 1]
    df["B"] = sRGB_clipped[:, 2]

    df["L*"] = Lab[:, 0]
    df["a*"] = Lab[:, 1]
    df["b*"] = Lab[:, 2]
    
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
            "x": gray_xyY[0, 0],
            "y": gray_xyY[0, 1],
            "Y_lum": gray_xyY[0, 2],
            "X": gray_xyz[0, 0],
            "Y": gray_xyz[0, 1],
            "Z": gray_xyz[0, 2],
            "R": gray_rgb[0, 0],
            "G": gray_rgb[0, 1],
            "B": gray_rgb[0, 2],
            "L*": avg_L,
            "a*": 0.0,
            "b*": 0.0,
            "HueDeg": 0,
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
            "x": xyY[0, 0],
            "y": xyY[0, 1],
            "Y_lum": xyY[0, 2],
            "X": xyz[0, 0],
            "Y": xyz[0, 1],
            "Z": xyz[0, 2],
            "R": rgb[0, 0],
            "G": rgb[0, 1],
            "B": rgb[0, 2],
            "L*": L,
            "a*": 0.0,
            "b*": 0.0,
            "HueDeg": 0,
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
def to_mesh(df_3d):
    # each entry of the dictionary is a df of all points with that Value
    # represents a horizontal "plate", which are stacked to form the space
    slices = dict(tuple(df_3d.groupby("Value")))
    
    for v in slices:
        # remove grayscale rows (Hue == "N") unless v == 0 or v == 10
        if v not in (0.0, 10.0):
            slices[v] = slices[v][slices[v]["Hue"] != "N"]
        # sort each slice by hue
        # only keep the highest chroma (outermost) vertex of each slice
        slices[v] = slices[v].sort_values("Chroma", ascending=False).drop_duplicates("HueDeg", keep="first")
        slices[v] = slices[v].sort_values("HueDeg")
    
    # 1-9 for munsell data, 0-10 including white and black
    values = sorted(slices.keys())
    
    vertices, faces = [], []
    index_map = {}
    global_index = 0
    
    # add all vertices to global vertices list
    for v in values:
        slice_df = slices[v]
        idx_list = []
        for _, row in slice_df.iterrows():
            vertices.append((row["X_3D"], Y_SCALE * row["Y_3D"], row["Z_3D"], row["R"], row["G"], row["B"]))
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

# Interpolates the Munsell renotation dataset along each of Hue, Value, Chroma
# perform the interpolation in CIELAB.
# original data set has: 
#   10 hue steps
#   11 value steps (inclusive of white and black)
#   28 maximum chroma
# target: 
#   70+ hue steps
#   20-30+ value steps
#   20-30+ chroma steps
def interpolate(df, steps_hue=8, steps_value=2, steps_chroma=1):
    """
    Interpolates extra points between existing Munsell data points.
    Works slice-by-slice (Hue/Chroma ring), between Value slices, and radially in Chroma.

    Parameters
    ----------
    df : DataFrame
        Processed dataframe with HueDeg, Value, Chroma, L*, a*, b*, R,G,B
    steps_hue : int
        Number of subdivisions between adjacent hue samples in a slice
    steps_value : int
        Number of subdivisions between adjacent Value slices
    steps_chroma : int
        Number of subdivisions between adjacent Chroma levels for same Hue/Value

    Returns
    -------
    DataFrame
        Combined dataframe with original and interpolated points, 
        with `is_original` flag.
    """
    df = df.copy()
    df["is_original"] = True
    all_points = []
    
    # interpolate circumferentially, between hues, within each Value "plate" (horizontal slice)
    for val, slice_df in df.groupby("Value"):
        slice_df = slice_df.sort_values("HueDeg")
        pts = slice_df[["HueDeg", "Chroma", "L*", "a*", "b*", "R", "G", "B"]].to_numpy()

        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i+1) % len(pts)]  # wrap around
            for t in np.linspace(0, 1, steps_hue+2)[1:-1]:  # skip endpoints
                interp = (1-t)*p1 + t*p2
                all_points.append({
                    "HueDeg": interp[0],
                    "Value": val,
                    "Chroma": interp[1],
                    "L*": interp[2], "a*": interp[3], "b*": interp[4],
                    "R": interp[5], "G": interp[6], "B": interp[7],
                    "is_original": False
                })
    
    df = pd.concat([df, pd.DataFrame(all_points)], ignore_index=True)
    all_points = []
    
    # interpolate vertically, between Value "plates"
    values = sorted(df["Value"].unique())
    for v1, v2 in zip(values[:-1], values[1:]):
        slice1 = df[df["Value"] == v1].sort_values("HueDeg").reset_index(drop=True)
        slice2 = df[df["Value"] == v2].sort_values("HueDeg").reset_index(drop=True)
        N = min(len(slice1), len(slice2))

        for i in range(N):
            p1 = slice1.iloc[i]
            p2 = slice2.iloc[i]
            for t in np.linspace(0, 1, steps_value+2)[1:-1]:
                interp = (1-t)*p1[["HueDeg","Chroma","L*","a*","b*","R","G","B"]].to_numpy() \
                         + t*p2[["HueDeg","Chroma","L*","a*","b*","R","G","B"]].to_numpy()
                all_points.append({
                    "HueDeg": interp[0],
                    "Value": (1-t)*p1["Value"] + t*p2["Value"],
                    "Chroma": interp[1],
                    "L*": interp[2], "a*": interp[3], "b*": interp[4],
                    "R": interp[5], "G": interp[6], "B": interp[7],
                    "is_original": False
                })

    df = pd.concat([df, pd.DataFrame(all_points)], ignore_index=True)
    all_points = []

    # interpolate radially along Chroma axis
    for (val, hue), group in df.groupby(["Value", "HueDeg"]):
        group = group.sort_values("Chroma")
        chroma_pts = group[["Chroma", "L*", "a*", "b*", "R", "G", "B"]].to_numpy()

        for i in range(len(chroma_pts) - 1):
            p1 = chroma_pts[i]
            p2 = chroma_pts[i+1]
            for t in np.linspace(0, 1, steps_chroma+2)[1:-1]:
                interp = (1-t)*p1 + t*p2
                all_points.append({
                    "HueDeg": hue,
                    "Value": val,
                    "Chroma": interp[0],
                    "L*": interp[1], "a*": interp[2], "b*": interp[3],
                    "R": interp[4], "G": interp[5], "B": interp[6],
                    "is_original": False
                })

    df = pd.concat([df, pd.DataFrame(all_points)], ignore_index=True)

    return df


# put all vertices in a point cloud
def to_pointcloud(df_3d):
    vertices = []
    for _, row in df_3d.iterrows():
        x, y, z = row["X_3D"], Y_SCALE * row["Y_3D"], row["Z_3D"]
        r, g, b = row["R"], row["G"], row["B"]
        vertices.append((x, y, z, r, g, b))
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
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for x, y, z, r, g, b in vertices:
            r_byte = int(r * 255)
            g_byte = int(g * 255)
            b_byte = int(b * 255)
            f.write(f"{x} {y} {z} {r_byte} {g_byte} {b_byte}\n")

        for face in faces:
            f.write(f"3 {' '.join(map(str, face))}\n")

def main():
    input_url = "https://www.rit-mcsl.org/MunsellRenotation/real.dat"
    df_raw = load_munsell_dat(input_url)
    
    df_processed = process(df_raw)
    df_processed.to_csv("munsell_parsed.csv", index=False)
    print("saved to munsell_parsed.csv")
    
    df_interpolated = interpolate(df_processed)
    df_interpolated.to_csv("munsell_interpolated.csv", index=False)
    print("saved to munsell_interpolated.csv")
    
    df_3d = to_3d_coordinates(df_interpolated)
    df_3d.to_csv("munsell_3d.csv", index=False)
    print("saved to munsell_3d.csv")
    
    # create a point cloud
    vertices = to_pointcloud(df_3d)
    write_ply(vertices, [], "munsell_pointcloud.ply")
    
    # create a "shell" mesh
    outer_vertices, faces = to_mesh(df_3d)
    write_ply(outer_vertices, faces, "munsell_mesh.ply")
    
    print(":)")
    


if __name__ == "__main__":
    main()
