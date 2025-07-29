import pandas as pd
import numpy as np
import re, urllib.request
from colour import xyY_to_XYZ, XYZ_to_sRGB, XYZ_to_Lab, CCS_ILLUMINANTS
from itertools import pairwise

Y_SCALE = 3

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
def munsell_hue_to_number(hue_str):
    hue_match = re.match(r'(\d{1,2}(\.\d+)?)([A-Z]+)', hue_str.strip())
    if not hue_match:
        return None
    number = float(hue_match.group(1))
    letter = hue_match.group(3)

    hue_order = ['R', 'YR', 'Y', 'GY', 'G', 'BG', 'B', 'PB', 'P', 'RP']
    base_index = hue_order.index(letter)

    return 3.6 * (base_index * 10 + number) # range: 0–360

def process(df):
    df["HueNumber"] = df["Hue"].apply(munsell_hue_to_number)

    # Convert xyY to XYZ
    xyY = df[["x", "y", "Y_lum"]].to_numpy()
    XYZ = xyY_to_XYZ(xyY)

    # Convert to sRGB (illuminant C used in renotation data)
    illuminant_C = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["C"]
    sRGB = XYZ_to_sRGB(XYZ, illuminant=illuminant_C)
    sRGB_clipped = np.clip(sRGB, 0, 1)
    # TODO: IF CLIPPED I NEED TO NOTE THAT

    # Convert to Lab
    Lab = XYZ_to_Lab(XYZ, illuminant=illuminant_C)

    df["X"] = XYZ[:, 0]
    df["Y"] = XYZ[:, 1]
    df["Z"] = XYZ[:, 2]

    df["R"] = sRGB_clipped[:, 0]
    df["G"] = sRGB_clipped[:, 1]
    df["B"] = sRGB_clipped[:, 2]

    df["L*"] = Lab[:, 0]
    df["a*"] = Lab[:, 1]
    df["b*"] = Lab[:, 2]
    
    # TODO ADD WHITE/BLACK AND GRAYSCALE VERTICES LOL

    return df

# Convert polar Hue/Chroma to X/Z, use Value as Y
def to_3d_coordinates(df):
    radians = np.deg2rad(df["HueNumber"])
    df["X_3D"] = df["Chroma"] * np.cos(radians)
    df["Y_3D"] = df["Value"]
    df["Z_3D"] = df["Chroma"] * np.sin(radians)
    return df

# generate 3d mesh defined by the outermost vertices
def to_mesh(df_3d):
    # each entry of the dictionary is a df of all points with that Value
    # represents a horizontal "plate", which are stacked to form the space
    slices = dict(tuple(df_3d.groupby("Value")))
    # sort each slice by hue
    # only keep the highest chroma vertex of each slice
    for v in slices:
        slices[v] = slices[v].sort_values("Chroma", ascending=False).drop_duplicates("HueNumber", keep="first")
        slices[v] = slices[v].sort_values("HueNumber")
    
    # should be 1-9
    values = sorted(slices.keys())
    
    vertices, faces = [], []
    index_map = {}
    global_index = 0
    
    
    
    # add all vertices to global vertices list
    for v in values:
        slice_df = slices[v]
        idx_list = []
        for _, row in slice_df.iterrows():
        # TODO: calculate the necessary scaling factor along Y axis 
        # that makes it look perceptually uniform
            vertices.append((row["X_3D"], Y_SCALE * row["Y_3D"], row["Z_3D"], row["R"], row["G"], row["B"]))
            idx_list.append(global_index)
            global_index += 1
        # and index them (will automatically go in by hue order)
        index_map[v] = idx_list

    # build faces between adjacent slices
    for v1, v2 in pairwise(values):
        idx1 = index_map[v1]
        idx2 = index_map[v2]
        N = min(len(idx1), len(idx2))

        for i in range(N):
            i_next = (i + 1) % N
            # form two triangles (that make 1 quad)
            faces.append((idx1[i], idx2[i], idx2[i_next]))
            faces.append((idx1[i], idx2[i_next], idx1[i_next]))
    
    return vertices, faces

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
    # df_processed.to_csv("munsell_parsed.csv", index=False)
    # print(f"saved to munsell_parsed.csv")
    
    df_3d = to_3d_coordinates(df_processed)
    # df_3d.to_csv("munsell_3d.csv", index=False)
    # print(f"saved to munsell_3d.csv")
    
    # create a point cloud
    vertices = to_pointcloud(df_3d)
    write_ply(vertices, [], "munsell_pointcloud.ply")
    print(":)")
    
    # create a "shell" mesh
    outer_vertices, faces = to_mesh(df_3d)
    write_ply(outer_vertices, faces, "munsell_mesh.ply")
    
    
    


if __name__ == "__main__":
    main()
