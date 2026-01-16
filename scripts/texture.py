import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from collections import defaultdict
from script import interpolate_preprocess, grid_interpolate, Lab_to_sRGB
import struct

# extends data by duplicating max chroma points outward to fill chroma space, up to the max of 38
def texture_preprocess(df, max_chroma=38):

    # duplicates grays and handles hue wrapping
    df = interpolate_preprocess(df)
    
    extended = []
    
    # Group by (hue, value) and extend chroma
    for (hue, value), group in df.groupby(["HueDeg", "Value"]):
        # find the maximum chroma point for this (hue, value)
        max_c_row = group.loc[group["Chroma"].idxmax()]
        actual_max_c = max_c_row["Chroma"]
        
        # duplicate this point at higher chroma values
        # spacing of 2 is the same as original munsell data
        for c in np.arange(actual_max_c + 2, max_chroma + 0.1, 2):
            extended_point = max_c_row.copy()
            extended_point["Chroma"] = c
            extended.append(extended_point)
    
    df_extended = pd.concat([df, pd.DataFrame(extended)], ignore_index=True)
    
    return df_extended

def to_3d_texture(df, h_res=128, v_res=32, c_res=64):
    # duplicate grays, wrap hue, extend chroma
    df_extended = texture_preprocess(df)
    
    # sort and make sure everything is unique for RegularGridInterpolator
    chroma_points = np.sort(df_extended['Chroma'].unique())
    hue_points = np.sort(df_extended['HueDeg'].unique())
    value_points = np.sort(df_extended['Value'].unique())

    # Build 3D grids for the data
    L_grid = np.zeros((len(chroma_points), len(hue_points), len(value_points)))
    a_grid = np.zeros((len(chroma_points), len(hue_points), len(value_points)))
    b_grid = np.zeros((len(chroma_points), len(hue_points), len(value_points)))

    # Fill the grids with data
    for idx, row in df_extended.iterrows():
        i = np.searchsorted(chroma_points, row['Chroma'])
        j = np.searchsorted(hue_points, row['HueDeg'])
        k = np.searchsorted(value_points, row['Value'])
        L_grid[i, j, k] = row['L*']
        a_grid[i, j, k] = row['a*']
        b_grid[i, j, k] = row['b*']

    interp_L = RegularGridInterpolator((chroma_points, hue_points, value_points), L_grid, bounds_error=False, fill_value=0)
    interp_a = RegularGridInterpolator((chroma_points, hue_points, value_points), a_grid, bounds_error=False, fill_value=0)
    interp_b = RegularGridInterpolator((chroma_points, hue_points, value_points), b_grid, bounds_error=False, fill_value=0)
    
    # Create texture coordinate grids
    chroma_grid = np.linspace(0, 38, c_res)
    hue_grid = np.linspace(0, 360, h_res)
    value_grid = np.linspace(0, 10, v_res)
    
    # Create meshgrid for vectorized interpolation
    C, H, V = np.meshgrid(chroma_grid, hue_grid, value_grid, indexing='ij')
    query_points = np.stack([C.ravel(), H.ravel(), V.ravel()], axis=1)
    
    #interpolate Lab values
    L_vals = interp_L(query_points).reshape(c_res, h_res, v_res)
    a_vals = interp_a(query_points).reshape(c_res, h_res, v_res)
    b_vals = interp_b(query_points).reshape(c_res, h_res, v_res)
    
    # Replace any remaining NaN with black
    L_vals = np.nan_to_num(L_vals, nan=0.0)
    a_vals = np.nan_to_num(a_vals, nan=0.0)
    b_vals = np.nan_to_num(b_vals, nan=0.0)
    
    # convert all to rgb
    lab_array = np.stack([L_vals.ravel(), a_vals.ravel(), b_vals.ravel()], axis=1)
    sRGB_array, is_clipped = Lab_to_sRGB(lab_array)
    
    # Reshape back to 3D
    r_vals = sRGB_array[:, 0].reshape(c_res, h_res, v_res)
    g_vals = sRGB_array[:, 1].reshape(c_res, h_res, v_res)
    b_vals = sRGB_array[:, 2].reshape(c_res, h_res, v_res)
    
    texture_data = {'R': r_vals, 'G': g_vals, 'B': b_vals}
    texture = texture_postprocess(texture_data)
    
    # print(f"dimensions: {texture.shape}")
    
    return texture


# reorder texture data for use with cylindrical shader sampling
# x = chroma => radius
# y = value => height (unchanged) (THREE.JS USES Y UP)
# z = hue => angle
def texture_postprocess(texture_data):  
    # [depth][height][width] = [hue][value][chroma]
    c_size, h_size, v_size = texture_data['R'].shape
    texture = np.zeros((h_size, v_size, c_size, 4), dtype=np.float32)
    
    # transpose from (chroma, hue, value) to (hue, value, chroma)
    texture[:, :, :, 0] = texture_data['R'].transpose(1, 2, 0)
    texture[:, :, :, 1] = texture_data['G'].transpose(1, 2, 0)
    texture[:, :, :, 2] = texture_data['B'].transpose(1, 2, 0)
    texture[:, :, :, 3] = 1.0
    
    return texture

def debug_textures(size=64):
    # white to black along x
    x_tex = np.zeros((size, size, size, 4), dtype=np.float32)
    for i in range(size):
        x_tex[i, :, :, :3] = i / (size - 1)
    x_tex[:, :, :, 3] = 1.0
    
    # y gradient
    y_tex = np.zeros((size, size, size, 4), dtype=np.float32)
    for j in range(size):
        y_tex[:, j, :, :3] = j / (size - 1)
    y_tex[:, :, :, 3] = 1.0
    
    # z gradient
    z_tex = np.zeros((size, size, size, 4), dtype=np.float32)
    for k in range(size):
        z_tex[:, :, k, :3] = k / (size - 1)
    z_tex[:, :, :, 3] = 1.0
    
    with open("debug_x.raw", 'wb') as f:
        x_tex.tofile(f)
    with open("debug_y.raw", 'wb') as f:
        y_tex.tofile(f)
    with open("debug_z.raw", 'wb') as f:
        z_tex.tofile(f)
    print("Created debug textures")    

def write_texture(texture, output_path):
    with open(output_path, 'wb') as f:
        texture.tofile(f)
    
    print(f"wrote to {output_path}")

if __name__ == "__main__":
    df = pd.read_csv("munsell_parsed.csv", index_col=False)
    texture = to_3d_texture(df)
    write_texture(texture, "munsell_texture.raw")
    
    