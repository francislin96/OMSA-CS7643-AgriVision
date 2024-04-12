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

# Values listed in R/G/B/NIR order
# These values are for the labeled training set ONLY
dataset_normalization = {
    "means": (111.45638762836, 113.896525896208, 112.225877823739, 118.304591896293),
    "std": (43.7544657087047, 41.2893581411199, 41.7538742005257, 46.5557582714383)
}