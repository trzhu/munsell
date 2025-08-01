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

# Lab to XYZ to RGB
def Lab_to_sRGB(lab):
    xyz = Lab_to_XYZ(lab, illuminant=ILLUM_C)
    sRGB = XYZ_to_sRGB(xyz, illuminant=ILLUM_C)
    sRGB_clipped = np.clip(sRGB, 0, 1)
    isClipped = sRGB == sRGB
    
    return sRGB_clipped, isClipped


def process(df):
    df["HueDeg"] = df["Hue"].apply(munsell_hue_to_deg)
        
    # munsell's data was recorded as xyY_lum, but Y_lum is scaled from 0-100 instead of 0-1
    xyY = df[["x", "y", "Y_lum"]].to_numpy()
    xyY[:, 2] /= 100  
    
    # Convert xyY to XYZ
    xyz = xyY_to_XYZ(xyY)

    sRGB = XYZ_to_sRGB(xyz, illuminant=ILLUM_C)
    sRGB_clipped = np.clip(sRGB, 0, 1)
    # TODO: IF CLIPPED should I write store that at the vertex or something?

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
#   40 hue steps
#   11 value steps (inclusive of white and black)
#   38 maximum chroma
# target: 
#   80+ hue steps
#   20-30+ value steps
#   30+ chroma steps
def interpolate(df, hue_steps=2, value_steps=2, chroma_steps=2):
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
    
    # interpolate radially along Chroma axis
    # splits the df into buckets that have the same value and huedeg
    # so we can interpolate between chroma
    for (value, hue), group in df.groupby(["Value", "HueDeg"]):
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
                sRGB, isClipped = Lab_to_sRGB(Lab)
                
                all_points.append({
                    "HueDeg": hue,
                    "Value": value,
                    "Chroma": chroma,
                    "L*": L, "a*": a, "b*": b,
                    "R": sRGB[0, 0], "G": sRGB[0, 1], "B": sRGB[0, 2],
                    "is_original": False
                })
    
    # drop the extra grayscales now that we're done with them
    df = df[~((df["Chroma"] == 0) & (df["HueDeg"] != 0))].reset_index(drop=True)
    
    df = pd.concat([df, pd.DataFrame(all_points)], ignore_index=True)
    all_points = []
    
    # interpolate vertically, between Value "plates"
    # TODO: if two layers have a different amount of vertices, the interpolated layer should have their avg
    # rn this just takes takes their minimum
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
                sRGB, isClipped = Lab_to_sRGB(Lab)
                
                all_points.append({
                    "HueDeg": hue,
                    "Value": value,
                    "Chroma": chroma,
                    "L*": L, "a*": a, "b*": b,
                    "R": sRGB[0, 0], "G": sRGB[0, 1], "B": sRGB[0, 2],
                    "is_original": False
                })

    df = pd.concat([df, pd.DataFrame(all_points)], ignore_index=True)    
    all_points = []
    
    # interpolate circumferentially, between hues, within each Value "plate" (horizontal slice)
    # za plan:
    # do each spoke one at a time again, doing two neighbouring axles at a time
    # work from the inside out again
    # if two spokes are different lengths, the interpolated spoke should again have their avg
    # TODO
    for (value, chroma), group in df.groupby(["Value", "Chroma"]):
        
        # skip grayscale
        if chroma == 0:
            continue
        if len(group) < 2:
            continue
                
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
                sRGB, isClipped = Lab_to_sRGB(Lab)
                
                all_points.append({
                    "HueDeg": hue,
                    "Value": value,
                    "Chroma": chroma,
                    "L*": L, "a*": a, "b*": b,
                    "R": sRGB[0, 0], "G": sRGB[0, 1], "B": sRGB[0, 2],
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
    # input_url = "https://www.rit-mcsl.org/MunsellRenotation/real.dat"
    # df_raw = load_munsell_dat(input_url)
    
    # df_processed = process(df_raw)
    # df_processed.to_csv("munsell_parsed.csv", index=False)
    # print("saved to munsell_parsed.csv")
    
    df_processed = pd.read_csv("munsell_parsed.csv", index_col=False)
    
    df_interpolated = interpolate(df_processed)
    df_interpolated.to_csv("munsell_interpolated.csv", index=False)
    print("saved to munsell_interpolated.csv")
    
    df_3d = to_3d_coordinates(df_interpolated)
    df_3d.to_csv("munsell_3d.csv", index=False)
    print("saved to munsell_3d.csv")
    
    # df_3d = pd.read_csv("munsell_3d.csv", index_col=False)
    
    # create a point cloud
    vertices = to_pointcloud(df_3d)
    write_ply(vertices, [], "munsell_pointcloud.ply")
    
    # create a "shell" mesh
    outer_vertices, faces = to_mesh(df_3d)
    write_ply(outer_vertices, faces, "munsell_mesh.ply")
    
    print(":)")
    


if __name__ == "__main__":
    main()
