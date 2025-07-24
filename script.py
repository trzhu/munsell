import pandas as pd
import numpy as np
from colour import xyY_to_XYZ, XYZ_to_sRGB, XYZ_to_Lab, CCS_ILLUMINANTS
import re
import urllib.request

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

# converts from munsell principal/adjacent hue system (RYGBP) to a number 0–100
def munsell_hue_to_number(hue_str):
    hue_match = re.match(r'(\d{1,2}(\.\d+)?)([A-Z]+)', hue_str.strip())
    if not hue_match:
        return None
    number = float(hue_match.group(1))
    letter = hue_match.group(3)

    hue_order = ['R', 'YR', 'Y', 'GY', 'G', 'BG', 'B', 'PB', 'P', 'RP']
    base_index = hue_order.index(letter)

    return (base_index * 10 + number)  # range: 0–100

def process(df):
    df["HueNumber"] = df["Hue"].apply(munsell_hue_to_number)

    # Convert xyY to XYZ
    xyY = df[["x", "y", "Y_lum"]].to_numpy()
    XYZ = xyY_to_XYZ(xyY)

    # Convert to sRGB (illuminant C used in renotation data)
    illuminant_C = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["C"]
    sRGB = XYZ_to_sRGB(XYZ, illuminant=illuminant_C)
    sRGB_clipped = np.clip(sRGB, 0, 1)

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

    return df

def main():
    input_url = "https://www.rit-mcsl.org/MunsellRenotation/real.dat"
    df_raw = load_munsell_dat(input_url)
    df_processed = process(df_raw)

    df_processed.to_csv("munsell_parsed.csv", index=False)
    print(f"saved to munsell_parsed.csv")

if __name__ == "__main__":
    main()
