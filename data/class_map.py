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
    "int_labs": [i for i in range(9)],
    "mask_vals": [0, 50, 75, 100, 125, 150, 175, 200, 255]
}
