 # segmentation class map

class_mapping = {
    "names": [
        "background",
        "double_plant",
        "drydown",
        "endrow",
        "nutrient_deficiency",
        "planter_skip",
        "water",
        "waterway",
        "weed_cluster"
    ],
    "int_labs": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "mask_vals": [0, 50, 75, 100, 125, 150, 175, 200, 255]
}

# Values listed in NIR/R/G/B format not NIR/B/G/R (more like opencv)
dataset_normalization = {
    "means": (0, 0, 0, 0),
    "std": (0, 0, 0, 0)
}