import json

# Load existing max_chroma data
with open('max_chroma.json', 'r') as f:
    max_chroma = json.load(f)

def get_max_chroma(hue, value):
    # black and white edge cases cases
    if value <= 0 or value >= 10:
        return 0
    
    # if already on the grid
    hue_key = str(int(hue))
    value_key = str(int(value))
    if hue_key in max_chroma and value_key in max_chroma[hue_key]:
        return max_chroma[hue_key][value_key]
    
    h_step = 9
    low_hue = ((int(hue) // h_step) * h_step) % 360
    high_hue = (low_hue + h_step) % 360
    
    low_value = int(value)
    high_value = min(10, low_value + 1)
    
    # get corner values - add .0 to match JSON format
    c00 = max_chroma.get(f"{low_hue}.0", {}).get(f"{low_value}.0", 0)
    c10 = max_chroma.get(f"{high_hue}.0", {}).get(f"{low_value}.0", 0)
    c01 = max_chroma.get(f"{low_hue}.0", {}).get(f"{high_value}.0", 0)
    c11 = max_chroma.get(f"{high_hue}.0", {}).get(f"{high_value}.0", 0)
    
    # Calculate weights
    if low_hue == 360 and high_hue == 9:
        if hue <= 180:
            hue_weight = (hue + 360 - 360) / (9 + 360 - 360)
        else:
            hue_weight = (hue - 360) / (9 + 360 - 360)
    else:
        hue_weight = (hue - low_hue) / (high_hue - low_hue)
    
    value_weight = (value - low_value) / (high_value - low_value)
    
    # bilerp
    c0 = c00 * (1 - hue_weight) + c10 * hue_weight
    c1 = c01 * (1 - hue_weight) + c11 * hue_weight
    result = c0 * (1 - value_weight) + c1 * value_weight
    
    return result

# find valid min and max V at a given hue and chroma
# march linearly in steps of 0.2 to find the closest value 
# the binary search the interval of size 0.2 to find the exact transition points
def find_value_range(h, c, min_v=0, max_v=10):
    EPSILON = 0.001
    STEP = 0.2
    
    valid_min_v = None
    valid_max_v = None
    
    # find validMinV - march upward
    v = min_v
    while v <= max_v and valid_min_v is None:
        max_chroma_at_v = get_max_chroma(h, v)
        if c <= max_chroma_at_v:
            # found first valid point! Binary search between v-STEP and v
            search_min = max(min_v, v - STEP)
            search_max = v
            
            while search_max - search_min > EPSILON:
                mid = (search_min + search_max) / 2
                if c <= get_max_chroma(h, mid):
                    search_max = mid  # valid, search lower
                else:
                    search_min = mid  # invalid, search higher
            
            valid_min_v = search_max
        else:
            v += STEP # march upward by 0.2
    
    # find validMaxV - march downward
    v = max_v
    while v >= min_v and valid_max_v is None:
        max_chroma_at_v = get_max_chroma(h, v)
        if c <= max_chroma_at_v:
            # found last valid point! Binary search between v and v+STEP
            search_min = v
            search_max = min(max_v, v + STEP)
            
            while search_max - search_min > EPSILON:
                mid = (search_min + search_max) / 2
                if c <= get_max_chroma(h, mid):
                    search_min = mid  # valid, search higher
                else:
                    search_max = mid  # invalid, search lower
            
            valid_max_v = search_min
        else:
            v -= STEP
    
    return valid_min_v, valid_max_v


# generate lookup table
value_ranges = {}

for hue in range(360):
    
    value_ranges[hue] = {}
    
    for chroma_index in range(76):  # 0 to 38 in steps of 0.5
        chroma = chroma_index * 0.5
        min_v, max_v = find_value_range(hue, chroma)
        
        # store as tuple [minV, maxV], with None values preserved
        value_ranges[hue][chroma] = [min_v, max_v]

with open('chroma_value_ranges.json', 'w') as f:
    json.dump(value_ranges, f, indent=2)

print("saved to chroma_value_ranges.json")