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

palette = {
    0 : (0, 0, 0),        # background
    1 : (255, 0, 0),    # double_plant
    2 : (255, 255, 0),    # drydown
    3 : (0, 255, 0),      # endrow
    4 : (0, 255, 255),      # nutrient_deficiency
    5 : (0, 0, 255),  # planter_skip
    6 : (255, 0, 255),    # water
    7 : (128, 128, 128),    # waterway
    8 : (255, 255, 255),    # weed_cluster
}

# Values listed in R/G/B/NIR order
# These values are for the labeled training set ONLY
dataset_normalization = {
    "means": (111.45638762836, 113.896525896208, 112.225877823739, 118.304591896293),
    "std": (43.7544657087047, 41.2893581411199, 41.7538742005257, 46.5557582714383)
}

labels_folder = {
    'double_plant': 1,
    'drydown': 2,
    'endrow': 3,
    'nutrient_deficiency': 4,
    'planter_skip': 5,
    'water': 6,
    'waterway': 7,
    'weed_cluster': 8
}

Data_Folder = {
    'Agriculture': {
        'ROOT': "./data/images_2021",
        'UNLABELED_ROOT': "./data/images_2024",
        'RGB': 'images/rgb/{}.jpg',
        'NIR': 'images/nir/{}.jpg',
        'SHAPE': (512, 512),
        'GT': 'gt/{}.png',
    },
}
palette_vsl = {
    0 : (0, 0, 0),        # background
    1 : (255, 0, 0),    # double_plant
    2 : (255, 255, 0),    # drydown
    3 : (0, 255, 0),      # endrow
    4 : (0, 255, 255),      # nutrient_deficiency
    5 : (0, 0, 255),  # planter_skip
    6 : (255, 0, 255),    # water
    7 : (128, 128, 128),    # waterway
    8 : (255, 255, 255),    # weed_cluster
}
land_classes = ["background", "double_plant", "drydown", "endrow","nutrient_deficiency",
                "planter_skip", "water", "waterway","weed_cluster"]