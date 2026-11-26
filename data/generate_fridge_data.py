"""
Generate a simulated 12-month usage history for a fridge management app.

The data is meant for training a scikit-learn classifier that predicts
whether a fridge item will be eaten or wasted.

Output files (written to the current working directory):
    fridge_history.json      one record per fridge item
    favorite_dishes.json     the favourite dishes of the simulated person
    products_catalogue.json  product data (shelf lives) an app would have
    DATA_README.md           documentation and summary statistics

Python 3.7 or newer, standard library only. Run:
    python generate_fridge_data.py
"""

import datetime
import json
import random


# ===========================================================================
# GENERAL SETTINGS
# ===========================================================================

# AI model that wrote this script (first line of DATA_README.md).
MODEL_NAME = "Claude Opus 5.5"

# File name of this script (regeneration instructions in DATA_README.md).
SCRIPT_NAME = "generate_fridge_data.py"

# Fixed random seed: the same seed always produces exactly the same data.
RANDOM_SEED = 42

# Simulated period: 12 months, simulated one day at a time.
START_DATE = datetime.date(2025, 9, 1)
END_DATE = datetime.date(2026, 8, 31)

# Output files.
HISTORY_FILE = "fridge_history.json"
DISHES_FILE = "favorite_dishes.json"
CATALOGUE_FILE = "products_catalogue.json"
README_FILE = "DATA_README.md"

# Optional input with real barcodes. If the file exists, catalogue products
# whose name matches an entry (upper/lower case ignored) get its barcode.
# Accepted formats:
#   [{"name": "Milk", "barcode": "6408430000258"}, ...]
#   {"6408430000258": {"name": "Milk"}, ...}
PRODUCTS_FILE = "products.json"

# Overall share of wasted items that the parameters were tuned for
# (reasoning in DATA_README.md).
TARGET_WASTE_RATE = (0.12, 0.20)


# ===========================================================================
# PRODUCT CATALOGUE
# ===========================================================================

# When a product is sold / bought: None = all year, otherwise a list of
# ((month, day), (month, day)) periods.
ALL_YEAR = None
GRILL_SEASON = [((5, 1), (8, 31))]
STRAWBERRY_SEASON = [((6, 15), (8, 15))]
NEW_POTATO_SEASON = [((6, 10), (8, 31))]
CHRISTMAS_SEASON = [((12, 1), (12, 31))]
HERRING_SEASON = [((6, 1), (6, 30)), ((12, 1), (12, 31))]

# Typical Finnish fridge products of a one-person household.
#   shelf    days from purchase to the date on the package (unopened)
#   opened   days the product stays good after the package is opened
#   portions uses per package (one coffee milk, one sandwich topping, one
#            yogurt bowl ...). Dishes and snacks use whole portions.
#   restock  probability of buying one package at the weekly shop when no
#            package with enough left is known to be at home (habit
#            products: at every weekly shop, see HABIT_PRODUCT_APPEAL);
#            0 = bought only for planned dishes
#   place    where the product is kept (see PLACE_FORGET_FACTOR)
#   season   see above
PRODUCTS = [
    # name                 category      shelf opened portions restock place      season
    ("Milk",               "dairy",        10,  6,  6, 0.95, "door",    ALL_YEAR),
    ("Yogurt",             "dairy",        18,  7,  4, 0.35, "shelf",   ALL_YEAR),
    ("Quark",              "dairy",        21,  3,  2, 0.20, "shelf",   ALL_YEAR),
    ("Sour cream",         "dairy",        18,  5,  2, 0.00, "shelf",   ALL_YEAR),
    ("Cooking cream",      "dairy",        30,  4,  2, 0.15, "door",    ALL_YEAR),
    ("Butter",             "dairy",        60, 45, 20, 0.95, "door",    ALL_YEAR),
    ("Cheese",             "dairy",        45, 28, 12, 0.90, "shelf",   ALL_YEAR),
    ("Cream cheese",       "dairy",        30,  7,  4, 0.10, "shelf",   ALL_YEAR),
    ("Eggs",               "eggs",         28, 28,  5, 0.75, "door",    ALL_YEAR),
    ("Minced meat",        "meat",          4,  1,  1, 0.00, "shelf",   ALL_YEAR),
    ("Chicken strips",     "meat",          6,  1,  1, 0.00, "shelf",   ALL_YEAR),
    ("Sliced ham",         "meat",         14,  5,  3, 0.30, "shelf",   ALL_YEAR),
    ("Ring sausage",       "meat",         21,  7,  3, 0.10, "shelf",   ALL_YEAR),
    ("Bacon",              "meat",         21,  4,  1, 0.00, "shelf",   ALL_YEAR),
    ("Grill sausages",     "meat",         18,  3,  2, 0.40, "shelf",   GRILL_SEASON),
    ("Christmas ham",      "meat",         14,  7,  4, 0.60, "shelf",   CHRISTMAS_SEASON),
    ("Salmon fillet",      "fish",          4,  1,  1, 0.00, "shelf",   ALL_YEAR),
    ("Cold-smoked salmon", "fish",         18,  4,  2, 0.15, "shelf",   ALL_YEAR),
    ("Pickled herring",    "fish",         30,  7,  3, 0.40, "back",    HERRING_SEASON),
    ("Rye bread",          "bread",         6,  3,  4, 0.80, "counter", ALL_YEAR),
    ("Toast bread",        "bread",         8,  6,  6, 0.10, "counter", ALL_YEAR),
    ("Karelian pies",      "bread",         5,  3,  3, 0.20, "counter", ALL_YEAR),
    ("Cucumber",           "vegetable",     8,  5,  4, 0.35, "drawer",  ALL_YEAR),
    ("Tomatoes",           "vegetable",     7,  7,  3, 0.25, "drawer",  ALL_YEAR),
    ("Lettuce",            "vegetable",     6,  5,  2, 0.25, "drawer",  ALL_YEAR),
    ("Carrots",            "vegetable",    30, 21,  6, 0.30, "drawer",  ALL_YEAR),
    ("Potatoes",           "vegetable",    21, 21,  5, 0.20, "drawer",  ALL_YEAR),
    ("New potatoes",       "vegetable",     7,  7,  3, 0.30, "drawer",  NEW_POTATO_SEASON),
    ("Bell pepper",        "vegetable",     9,  3,  2, 0.25, "drawer",  ALL_YEAR),
    ("Broccoli",           "vegetable",     6,  5,  1, 0.05, "drawer",  ALL_YEAR),
    ("Leek",               "vegetable",    14, 10,  2, 0.00, "drawer",  ALL_YEAR),
    ("Strawberries",       "fruit",         3,  2,  2, 0.60, "drawer",  STRAWBERRY_SEASON),
    ("Grapes",             "fruit",        10,  5,  2, 0.25, "drawer",  ALL_YEAR),
    ("Liver casserole",    "ready_meal",   10,  3,  1, 0.15, "shelf",   ALL_YEAR),
    ("Rosolli",            "ready_meal",   10,  4,  2, 0.60, "shelf",   CHRISTMAS_SEASON),
    ("Carrot casserole",   "ready_meal",   12,  4,  2, 0.60, "shelf",   CHRISTMAS_SEASON),
    ("Lingonberry jam",    "condiment",   120, 45,  6, 0.90, "back",    ALL_YEAR),
]


# ===========================================================================
# DAILY ROUTINE
# ===========================================================================
# Monday-Friday lunch is eaten at work (staff restaurant), Saturday and
# Sunday lunch at home.

# Coffee with milk in the morning (one portion of milk).
P_COFFEE_MILK = 0.85

# Breakfast at home (otherwise skipped).
P_BREAKFAST = 0.9

# Afternoon snack.
P_AFTERNOON_SNACK = 0.6

# Late-evening snack on normal evenings and on weekend evenings
# (WEEKEND_EVENINGS: 4 = Friday, 5 = Saturday).
P_LATE_SNACK_WEEKDAY = 0.2
P_LATE_SNACK_WEEKEND = 0.35
WEEKEND_EVENINGS = [4, 5]

# Dinner on a normal day: leftovers first (see LEFTOVERS and USE_UP_DAYS),
# otherwise cook a favourite dish with this probability, Monday ... Sunday.
# Fridays are often takeaway or eating out.
COOK_PROBABILITY = [0.65, 0.65, 0.6, 0.6, 0.35, 0.6, 0.7]

# Not cooking (or cooking was not possible): probability of eating out
# instead of a snack dinner at home, Monday ... Sunday.
EAT_OUT_PROBABILITY = [0.1, 0.1, 0.1, 0.15, 0.4, 0.3, 0.1]


# ===========================================================================
# SNACKS (eating without cooking a favourite dish)
# ===========================================================================

# Christmas foods, eaten as a "festive plate" (pickled herring also at
# Midsummer).
CHRISTMAS_FOODS = [
    "Christmas ham", "Rosolli", "Carrot casserole", "Pickled herring",
]

# A snack template lists slots: (products or categories to choose from,
# probability of using the slot). The first slot is required: a template is
# possible only when something for it is in the fridge. Each used slot takes
# one portion of one item.
SNACK_TEMPLATES = {
    "sandwich": [
        (["bread"], 1.0),
        (["Butter"], 0.8),
        (["Cheese", "Cream cheese"], 0.8),
        (["Sliced ham", "Cold-smoked salmon"], 0.7),
        (["Cucumber", "Tomatoes", "Bell pepper", "Lettuce"], 0.8),
    ],
    "porridge": [
        (["Milk"], 1.0),
        (["Lingonberry jam", "Strawberries"], 0.6),
    ],
    "yogurt bowl": [
        (["Yogurt", "Quark"], 1.0),
        (["Strawberries", "Grapes", "Lingonberry jam"], 0.4),
    ],
    "eggs": [
        (["Eggs"], 1.0),
        (["bread"], 0.7),
        (["Butter"], 0.5),
    ],
    "karelian pies": [
        (["Karelian pies"], 1.0),
        (["Butter"], 0.6),
        (["Eggs"], 0.3),
    ],
    "fruit": [
        (["fruit"], 1.0),
    ],
    "raw vegetables": [
        (["Carrots", "Cucumber", "Bell pepper", "Tomatoes"], 1.0),
    ],
    "ready meal": [
        (["Liver casserole"], 1.0),
        (["Lingonberry jam"], 0.5),
    ],
    "sausage": [
        (["Ring sausage", "Grill sausages"], 1.0),
        (["bread"], 0.4),
    ],
    "festive plate": [
        (CHRISTMAS_FOODS, 1.0),
        (CHRISTMAS_FOODS, 0.7),
        (["Potatoes", "New potatoes"], 0.5),
    ],
    "leftovers": [
        (["leftovers"], 1.0),
    ],
}

# What to eat: templates, and products within a slot, are chosen with
# weights that grow when an item for them is already opened (people finish
# open packages) or close to its last good day (see USE_UP_FACTOR).
OPENED_PREFERENCE_FACTOR = 3.0

# Relative weights of the templates on each eating occasion (only possible
# templates are considered).
BREAKFAST_TEMPLATES = {
    "sandwich": 5,
    "porridge": 2,
    "yogurt bowl": 3,
    "eggs": 1,
    "karelian pies": 1,
}
WEEKEND_LUNCH_TEMPLATES = {
    "sandwich": 3,
    "eggs": 2,
    "karelian pies": 1,
    "ready meal": 2,
    "sausage": 1,
    "festive plate": 4,
}
AFTERNOON_SNACK_TEMPLATES = {
    "fruit": 7,
    "raw vegetables": 3,
    "sandwich": 3,
    "yogurt bowl": 3,
    "karelian pies": 2,
}
SNACK_DINNER_TEMPLATES = {
    "ready meal": 4,
    "sandwich": 3,
    "eggs": 2,
    "sausage": 3,
    "festive plate": 6,
}
LATE_SNACK_TEMPLATES = {
    "sandwich": 4,
    "leftovers": 3,
    "yogurt bowl": 2,
    "sausage": 1,
    "fruit": 3,
    "festive plate": 3,
}


# ===========================================================================
# FAVOURITE DISHES AND COOKING
# ===========================================================================
# name, rating (1-5), season ("all", "winter" or "summer"), servings, and
# ingredients as (product, portions). The first ingredient is the key
# ingredient; its amount is one whole package, so every cooking uses up at
# least one item (this is how the dish shows in fridge_history.json).
FAVORITE_DISHES = [
    ("Macaroni casserole", 5, "all", 4,
     [("Minced meat", 1), ("Milk", 3), ("Eggs", 1)]),
    ("Salmon soup", 5, "winter", 3,
     [("Salmon fillet", 1), ("Potatoes", 2), ("Carrots", 1), ("Leek", 1),
      ("Cooking cream", 2)]),
    ("Tacos", 5, "all", 2,
     [("Minced meat", 1), ("Tomatoes", 1), ("Cucumber", 1), ("Lettuce", 1),
      ("Sour cream", 2), ("Cheese", 1)]),
    ("Minced meat sauce with potatoes", 4, "winter", 3,
     [("Minced meat", 1), ("Potatoes", 2), ("Cooking cream", 1),
      ("Carrots", 1)]),
    ("Spaghetti bolognese", 4, "all", 3,
     [("Minced meat", 1), ("Carrots", 1), ("Bell pepper", 1), ("Cheese", 1)]),
    ("Creamy chicken pasta", 4, "all", 2,
     [("Chicken strips", 1), ("Cooking cream", 2), ("Broccoli", 1),
      ("Cheese", 1)]),
    ("Chicken salad", 3, "summer", 1,
     [("Chicken strips", 1), ("Lettuce", 2), ("Cucumber", 1), ("Tomatoes", 1),
      ("Bell pepper", 1)]),
    ("Oven salmon with potatoes", 4, "all", 2,
     [("Salmon fillet", 1), ("Potatoes", 2), ("Broccoli", 1)]),
    ("Sausage soup", 3, "winter", 3,
     [("Ring sausage", 3), ("Potatoes", 2), ("Carrots", 2), ("Leek", 1)]),
    ("Pasta carbonara", 4, "all", 2,
     [("Bacon", 1), ("Eggs", 1), ("Cooking cream", 1), ("Cheese", 1)]),
    ("Oven pancake", 3, "all", 3,
     [("Milk", 6), ("Eggs", 2), ("Lingonberry jam", 1)]),
    ("Meatballs with mashed potatoes", 4, "winter", 3,
     [("Minced meat", 1), ("Potatoes", 3), ("Milk", 2), ("Eggs", 1),
      ("Lingonberry jam", 1)]),
    ("Vegetable soup", 3, "winter", 2,
     [("Cooking cream", 2), ("Carrots", 2), ("Potatoes", 2), ("Broccoli", 1),
      ("Leek", 1), ("Bell pepper", 1)]),
    ("Grilled sausages with new potatoes", 4, "summer", 1,
     [("Grill sausages", 2), ("New potatoes", 3), ("Cucumber", 1)]),
    ("Chicken curry", 4, "all", 2,
     [("Chicken strips", 1), ("Cooking cream", 1), ("Bell pepper", 1)]),
]

# A dish can be cooked when its key ingredient is in the fridge in full and
# at least this share of the other ingredients is available (missing ones
# are left out). If the key ingredient is about to spoil (USE_UP_DAYS), the
# person improvises and cooks the dish even without the other ingredients.
MIN_OTHER_INGREDIENTS_SHARE = 0.5

# Choice weight of a dish = rating x season factor x variety factor
# x use-up factor.
WINTER_MONTHS = [10, 11, 12, 1, 2, 3]
SUMMER_MONTHS = [6, 7, 8]
DISH_IN_SEASON_FACTOR = 1.5
DISH_OUT_OF_SEASON_FACTOR = 0.5
# Variety: a dish cooked during the last RECENTLY_COOKED_DAYS days is less
# likely.
RECENTLY_COOKED_DAYS = 6
RECENTLY_COOKED_FACTOR = 0.2
# Using things up: a dish is more likely when its key ingredient is at most
# USE_UP_DAYS days from its last good day (snacks: when any item for them
# is). Then the person also cooks before eating leftovers, with at least
# P_COOK_WHEN_USING_UP.
USE_UP_DAYS = 2
USE_UP_FACTOR = 5.0
P_COOK_WHEN_USING_UP = 0.85

# Side vegetables: when cooking, one portion of a vegetable (the one the
# person would pick, e.g. an opened one) is eaten as a side or added to the
# dish with this probability.
P_SIDE_VEGETABLE = 0.8


# ===========================================================================
# LEFTOVERS
# ===========================================================================
# One serving is eaten right after cooking; the rest (servings - 1, +-
# LEFTOVER_PORTIONS_NOISE) goes into one leftovers box. Leftovers can be
# eaten from the next day on and spoil after LEFTOVERS_SHELF_LIFE days.
LEFTOVERS_SHELF_LIFE = 3

# Base probabilities of eating leftovers when there are some.
P_LUNCHBOX = 0.8           # Monday-Friday lunch at work
P_LEFTOVERS_LUNCH = 0.7    # Saturday-Sunday lunch at home
P_LEFTOVERS_DINNER = 0.7   # dinner

# The base probability is multiplied by rating / LEFTOVERS_RATING_REFERENCE
# (favourite dishes are eaten more gladly) and by LEFTOVERS_FATIGUE for
# every day after the first (the same food gets boring).
LEFTOVERS_RATING_REFERENCE = 4
LEFTOVERS_FATIGUE = 0.85


# ===========================================================================
# SHOPPING
# ===========================================================================

# Day of the bigger weekly shop, chosen every Monday (0 = Monday ...
# 6 = Sunday): mostly Friday or the weekend. The simulation starts with an
# empty fridge and a big shop on the first day.
BIG_SHOP_DAY_WEIGHTS = {3: 0.05, 4: 0.50, 5: 0.30, 6: 0.15}

# Number of favourite dishes planned for the coming week (random integer in
# this range, planned dishes have different key ingredients). Missing
# ingredients are bought in whole packages.
PLANNED_DISHES_PER_WEEK = (2, 3)

# Probability of checking the fridge before the weekly shop. Otherwise the
# person relies on memory and remembers each item with P_REMEMBER_ITEM, so
# products that are already at home get bought again.
P_CHECK_FRIDGE = 0.8
P_REMEMBER_ITEM = 0.75

# A package counts as "enough at home" when at least this share of its
# portions is left.
ENOUGH_LEFT_SHARE = 0.5

# Trips: when a trip starts within TRIP_PLANNING_DAYS days, the person
# remembers it at the weekly shop with P_REMEMBER_TRIP; a holiday of at
# least LONG_TRIP_DAYS days is planned long before and always remembered.
# Then at most one dish per TRIP_DAYS_PER_DISH days before the trip is
# planned, and nothing is restocked if the trip starts within
# TRIP_NO_RESTOCK_DAYS days. No top-up trips within those days either.
TRIP_PLANNING_DAYS = 7
P_REMEMBER_TRIP = 0.7
LONG_TRIP_DAYS = 7
TRIP_DAYS_PER_DISH = 2
TRIP_NO_RESTOCK_DAYS = 3

# Small top-up trips on other days: if nothing matching the key (product or
# category) is found in the fridge, the person buys the given product with
# probability P_TOPUP_WHEN_OUT.
TOPUP_ESSENTIALS = {"Milk": "Milk", "bread": "Rye bread"}
P_TOPUP_WHEN_OUT = 0.5

# Wanting to cook but no favourite dish can be made: shop for one dish with
# this probability (otherwise eating out or a snack dinner).
P_TOPUP_FOR_DISH = 0.4

# Overbuying.
# Special offer ("3 for 2") at the weekly shop: extra packages of a random
# regularly bought product (restock > 0).
P_SPECIAL_OFFER = 0.15
OFFER_EXTRA_PACKAGES = (1, 2)

# Shopping while hungry (any shop): extra impulse items.
P_SHOPPING_HUNGRY = 0.15
HUNGRY_EXTRA_ITEMS = (1, 2)
IMPULSE_PRODUCTS = [
    "Karelian pies", "Cold-smoked salmon", "Cream cheese", "Grapes",
    "Quark", "Toast bread", "Strawberries", "Grill sausages",
]

# Red-label discount (any shop): one product close to its date, with only
# a few days left.
P_RED_LABEL = 0.1
RED_LABEL_DAYS_LEFT = (1, 2)
RED_LABEL_PRODUCTS = [
    "Minced meat", "Chicken strips", "Salmon fillet", "Sliced ham",
    "Rye bread", "Karelian pies", "Milk",
]


# ===========================================================================
# TRAVEL AND BUSY WEEKS
# ===========================================================================
# Away from home (first day, last day, reason). Nothing is eaten or bought;
# items that spoil meanwhile are thrown away after coming home.
TRAVEL_PERIODS = [
    (datetime.date(2025, 10, 17), datetime.date(2025, 10, 19),
     "weekend visit to friends"),
    (datetime.date(2025, 12, 23), datetime.date(2025, 12, 27),
     "Christmas with family"),
    (datetime.date(2026, 2, 21), datetime.date(2026, 2, 28),
     "ski holiday week"),
    (datetime.date(2026, 4, 3), datetime.date(2026, 4, 6),
     "Easter with family"),
    (datetime.date(2026, 6, 19), datetime.date(2026, 6, 21),
     "Midsummer at a summer cottage"),
    (datetime.date(2026, 7, 6), datetime.date(2026, 7, 19),
     "summer holiday trip"),
    (datetime.date(2026, 8, 7), datetime.date(2026, 8, 9),
     "summer cottage weekend"),
]

# Busy weeks at home (first day, last day, reason): no cooking. They are not
# expected when shopping, so food bought for the week tends to spoil.
BUSY_PERIODS = [
    (datetime.date(2025, 11, 17), datetime.date(2025, 11, 26),
     "work deadline"),
    (datetime.date(2026, 1, 26), datetime.date(2026, 2, 1),
     "busy week at work"),
    (datetime.date(2026, 3, 16), datetime.date(2026, 3, 25),
     "work deadline"),
    (datetime.date(2026, 5, 11), datetime.date(2026, 5, 17),
     "busy week at work"),
]

# On busy days dinner is eaten out with this probability, otherwise
# leftovers or a snack dinner.
P_EAT_OUT_BUSY = 0.8

# On busy days the person is rushed: breakfast and the afternoon snack at
# home are less likely (probabilities multiplied by this factor), and items
# are overlooked more often (FORGET_PROBABILITY multiplied by this factor).
BUSY_HOME_SNACK_FACTOR = 0.5
BUSY_FORGET_FACTOR = 3.0


# ===========================================================================
# SEASONAL EFFECTS
# ===========================================================================
# Summer (SUMMER_MONTHS): less cooking and more eating out.
SUMMER_COOK_FACTOR = 0.8
SUMMER_EAT_OUT_FACTOR = 1.3

# Christmas period ((month, day) range): every Christmas food restocked gets
# one extra package with P_CHRISTMAS_EXTRA_PACKAGE, and less is cooked.
CHRISTMAS_PERIOD = ((12, 10), (12, 31))
P_CHRISTMAS_EXTRA_PACKAGE = 0.5
CHRISTMAS_COOK_FACTOR = 0.6


# ===========================================================================
# WASTE TENDENCIES
# ===========================================================================
# Every morning each item may be overlooked for the whole day ("out of
# sight, out of mind"): it is not eaten and not seen when checking the
# fridge. Base probability per category (how much attention this kind of
# food gets; meat, fish and eggs are bought for planned meals and rarely
# forgotten). It is multiplied by the place and personal factors below.
FORGET_PROBABILITY = {
    "vegetable": 0.05,
    "fruit": 0.05,
    "leftovers": 0.03,
    "ready_meal": 0.06,
    "dairy": 0.05,
    "bread": 0.05,
    "condiment": 0.02,
    "meat": 0.02,
    "fish": 0.02,
    "eggs": 0.02,
}

# Opened packages are at the front and forgotten less often.
OPENED_FORGET_FACTOR = 0.5

# Taking an item: first the product is chosen (see
# OPENED_PREFERENCE_FACTOR), then among packages of it an opened one with
# P_TAKE_OPENED_FIRST, then the one closest to its last good day with
# P_TAKE_OLDEST_FIRST, otherwise a random one (e.g. the newest at the
# front). So with similar items in the fridge, older ones often spoil.
P_TAKE_OPENED_FIRST = 0.85
P_TAKE_OLDEST_FIRST = 0.7

# Spoiled items are thrown away the day after their last good day, or
# noticed late (P_NOTICE_LATE) and thrown away LATE_NOTICE_DAYS after it.
P_NOTICE_LATE = 0.35
LATE_NOTICE_DAYS = (2, 6)


# ===========================================================================
# PERSONAL HABITS
# ===========================================================================
# Stable habits of this person, the same all year. A fridge app could learn
# their effect from the history (e.g. "you often waste leek"), but it does
# not know them directly: they are not in the output files.

# Where a product is kept (column "place" in PRODUCTS). Food in sight gets
# used; food pushed to the back or into the crisper drawer is forgotten.
# Multiplies the daily probability of overlooking an item.
PLACE_FORGET_FACTOR = {
    "door": 0.5,     # door shelves: seen every time the fridge is opened
    "shelf": 1.0,    # front of a shelf
    "counter": 1.0,  # bread bin on the kitchen counter
    "drawer": 1.6,   # crisper drawer below the shelves, out of sight
    "back": 2.0,     # back of a shelf, behind other food (jars, boxes)
}

# Leftovers boxes go wherever there is room, usually behind other food.
LEFTOVERS_PLACE = "back"

# Personal blind spots: multiplies the daily probability of overlooking
# the product (products not listed: 1).
PERSONAL_FORGET_FACTOR = {
    # Daily staples, used every day: never overlooked.
    "Milk": 0.0,
    "Butter": 0.0,
    "Rye bread": 0.0,
    # Bought for one dish, the rest is often forgotten: leek under the
    # other vegetables, cream cheese and sour cream behind bigger packages.
    "Leek": 2.0,
    "Cream cheese": 2.0,
    "Sour cream": 2.0,
}

# Habit purchases: products bought at every weekly shop with their restock
# probability out of routine or good intentions ("I should eat more
# healthily"), even when there is still some at home, but rarely chosen
# when something tastier is available. Value = appeal when choosing what
# to eat (1 = normal; products not listed: 1).
HABIT_PRODUCT_APPEAL = {
    "Yogurt": 0.5,   # healthy breakfast plan, but sandwiches win
    "Lettuce": 0.5,  # salad plan
    "Grapes": 0.5,   # healthy snack plan
}


# ===========================================================================
# NOISE
# ===========================================================================
# Remaining shelf life at purchase varies by +- this share of the shelf
# life (rounded, at least 1 day), because shop stock is of different age.
SHELF_LIFE_NOISE_SHARE = 0.15

# Opened shelf life varies by +- this many days (at least 1 day).
OPENED_LIFE_NOISE_DAYS = 1

# Leftover portions vary by +- this many portions.
LEFTOVER_PORTIONS_NOISE = 1


ONE_DAY = datetime.timedelta(days=1)


# ===========================================================================
# CATALOGUE, DISHES AND BARCODES
# ===========================================================================

def build_catalogue():
    """Turn the PRODUCTS table into product dicts keyed by name."""
    catalogue = {}
    for row in PRODUCTS:
        name, category, shelf, opened, portions, restock, place, season = row
        catalogue[name] = {
            "name": name,
            "category": category,
            "shelf_life": shelf,
            "opened_life": opened,
            "portions": portions,
            "restock": restock,
            "place": place,
            "season": season,
            "barcode": None,
        }
    return catalogue


def build_dishes():
    """Turn the FAVORITE_DISHES table into dish dicts with ids 1, 2, ..."""
    dishes = []
    dish_id = 1
    for name, rating, season, servings, ingredients in FAVORITE_DISHES:
        dishes.append({
            "id": dish_id,
            "name": name,
            "rating": rating,
            "season": season,
            "servings": servings,
            "ingredients": ingredients,
            "leftovers_barcode": None,
        })
        dish_id += 1
    return dishes


def ean13_check_digit(first_12_digits):
    """Check digit of an EAN-13 code: weights 1, 3, 1, 3, ... from the left."""
    total = 0
    for index in range(12):
        digit = int(first_12_digits[index])
        if index % 2 == 0:
            total += digit
        else:
            total += 3 * digit
    return str((10 - total % 10) % 10)


def fake_barcode(used_codes):
    """Unique fake EAN-13 code in the store-internal range 20-29."""
    while True:
        code = "2" + str(random.randint(0, 9))
        for _ in range(10):
            code += str(random.randint(0, 9))
        code += ean13_check_digit(code)
        if code not in used_codes:
            used_codes.add(code)
            return code


def load_real_barcodes():
    """Read {lower-case product name: barcode} from PRODUCTS_FILE.
    Returns an empty dict if the file does not exist."""
    try:
        with open(PRODUCTS_FILE, encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return {}

    entries = []
    if isinstance(data, dict):
        for barcode, value in data.items():
            if isinstance(value, dict):
                entries.append({"barcode": barcode, "name": value.get("name")})
            else:
                entries.append({"barcode": barcode, "name": value})
    else:
        entries = data

    barcodes = {}
    for entry in entries:
        name = entry.get("name")
        barcode = entry.get("barcode")
        if name and barcode:
            barcodes[str(name).strip().lower()] = str(barcode)
    return barcodes


def assign_barcodes(catalogue, dishes):
    """Give every product and every dish's leftovers a barcode. Returns how
    many products got a real barcode. Fake codes are always drawn first, so
    products.json does not change the random numbers of the simulation."""
    used_codes = set()
    for product in catalogue.values():
        product["barcode"] = fake_barcode(used_codes)
    for dish in dishes:
        dish["leftovers_barcode"] = fake_barcode(used_codes)

    real_barcodes = load_real_barcodes()
    real_count = 0
    for product in catalogue.values():
        key = product["name"].lower()
        if key in real_barcodes:
            product["barcode"] = real_barcodes[key]
            real_count += 1
    return real_count


# ===========================================================================
# CALENDAR HELPERS
# ===========================================================================

def in_season(product, day):
    """True if the product is sold on this day."""
    if product["season"] is None:
        return True
    for start, end in product["season"]:
        if start <= (day.month, day.day) <= end:
            return True
    return False


def in_christmas_period(day):
    start, end = CHRISTMAS_PERIOD
    return start <= (day.month, day.day) <= end


def in_periods(day, periods):
    for first_day, last_day, reason in periods:
        if first_day <= day <= last_day:
            return True
    return False


def next_trip(day):
    """(days until the next trip starts, its length in days), or
    (None, None) if there are no more trips."""
    nearest = None
    length = None
    for first_day, last_day, reason in TRAVEL_PERIODS:
        if first_day > day:
            days = (first_day - day).days
            if nearest is None or days < nearest:
                nearest = days
                length = (last_day - first_day).days + 1
    return nearest, length


def choose_big_shop_day(monday):
    weekdays = list(BIG_SHOP_DAY_WEIGHTS.keys())
    weights = list(BIG_SHOP_DAY_WEIGHTS.values())
    weekday = random.choices(weekdays, weights)[0]
    return monday + datetime.timedelta(days=weekday)


# ===========================================================================
# FRIDGE ITEMS
# ===========================================================================

def new_item(state, name, category, barcode, place, day, days_left,
             portions, opened_life):
    """Create an item, put it in the fridge and return it."""
    item = {
        "name": name,
        "category": category,
        "barcode": barcode,
        "place": place,
        "added_date": day,
        "expiry_date": day + datetime.timedelta(days=days_left),
        "opened_date": None,
        "removed_date": None,
        "outcome": None,
        "consumed_by": None,
        "dish_id": None,
        "portions_left": portions,
        "opened_life": opened_life,
        "visible": True,
        "available_from": day,
        "spoiled": False,
        "removal_day": None,
        "source_dish": None,
    }
    state["fridge"].append(item)
    state["items"].append(item)
    return item


def buy(state, day, product, packages, red_label=False):
    """Buy packages of a product. Returns the new items."""
    bought = []
    for _ in range(packages):
        if red_label:
            days_left = random.randint(RED_LABEL_DAYS_LEFT[0],
                                       RED_LABEL_DAYS_LEFT[1])
        else:
            noise = max(1, round(product["shelf_life"] * SHELF_LIFE_NOISE_SHARE))
            days_left = product["shelf_life"] + random.randint(-noise, noise)
            days_left = max(1, days_left)
        opened_life = product["opened_life"] + random.randint(
            -OPENED_LIFE_NOISE_DAYS, OPENED_LIFE_NOISE_DAYS)
        opened_life = max(1, opened_life)
        item = new_item(state, product["name"], product["category"],
                        product["barcode"], product["place"], day, days_left,
                        product["portions"], opened_life)
        bought.append(item)
    return bought


def add_leftovers(state, day, dish, portions):
    item = new_item(state, dish["name"] + " (leftovers)", "leftovers",
                    dish["leftovers_barcode"], LEFTOVERS_PLACE, day,
                    LEFTOVERS_SHELF_LIFE, portions, LEFTOVERS_SHELF_LIFE)
    item["available_from"] = day + ONE_DAY
    item["source_dish"] = dish


def last_good_day(item):
    """The date on the package, or the end of the opened shelf life if that
    comes first."""
    last_day = item["expiry_date"]
    if item["opened_date"] is not None:
        opened_limit = item["opened_date"] + datetime.timedelta(
            days=item["opened_life"])
        if opened_limit < last_day:
            last_day = opened_limit
    return last_day


def remove_item(state, item, day, outcome, consumed_by, dish_id):
    item["removed_date"] = day
    item["outcome"] = outcome
    item["consumed_by"] = consumed_by
    item["dish_id"] = dish_id
    state["fridge"].remove(item)


def usable_items(state, day, wanted):
    """Items matching a product name or a category that the person sees
    today and can still eat."""
    found = []
    for item in state["fridge"]:
        if item["name"] != wanted and item["category"] != wanted:
            continue
        if item["spoiled"] or not item["visible"]:
            continue
        if item["available_from"] > day:
            continue
        found.append(item)
    return found


def total_portions(items):
    total = 0
    for item in items:
        total += item["portions_left"]
    return total


def appeal(name):
    """How gladly the person eats a product (1 = normal)."""
    return HABIT_PRODUCT_APPEAL.get(name, 1.0)


def preference_factor(items, day):
    """Extra weight for eating something: an opened package, or an item
    close to its last good day."""
    opened = False
    expiring = False
    for item in items:
        if item["opened_date"] is not None:
            opened = True
        if (last_good_day(item) - day).days <= USE_UP_DAYS:
            expiring = True
    factor = 1.0
    if opened:
        factor *= OPENED_PREFERENCE_FACTOR
    if expiring:
        factor *= USE_UP_FACTOR
    return factor


def choose_item(candidates, day):
    """The item the person takes when several fit: first the product (by
    preference), then among packages of it an opened one, then the one
    closest to its last good day, otherwise a random one."""
    products = {}
    for item in candidates:
        if item["name"] not in products:
            products[item["name"]] = []
        products[item["name"]].append(item)
    names = []
    weights = []
    for name in products:
        names.append(name)
        weights.append(preference_factor(products[name], day) * appeal(name))
    items = products[random.choices(names, weights)[0]]

    opened = []
    for item in items:
        if item["opened_date"] is not None:
            opened.append(item)
    if opened and random.random() < P_TAKE_OPENED_FIRST:
        items = opened

    if random.random() < P_TAKE_OLDEST_FIRST:
        oldest = items[0]
        for item in items:
            if last_good_day(item) < last_good_day(oldest):
                oldest = item
        return oldest
    return random.choice(items)


def take_portions(state, item, amount, day, consumed_by, dish_id):
    """Use up to `amount` portions of one item. Returns the portions used."""
    used = min(amount, item["portions_left"])
    item["portions_left"] -= used
    if item["opened_date"] is None:
        item["opened_date"] = day
    if item["portions_left"] == 0:
        if item["category"] == "leftovers":
            # Eating leftovers means eating the dish they came from.
            consumed_by = "dish"
            dish_id = item["source_dish"]["id"]
        remove_item(state, item, day, "eaten", consumed_by, dish_id)
    return used


def use_portions(state, day, wanted, amount, consumed_by, dish_id):
    """Use `amount` portions of a product or category, taking items in the
    order the person would. Returns the portions used."""
    used = 0
    while used < amount:
        candidates = usable_items(state, day, wanted)
        if not candidates:
            break
        item = choose_item(candidates, day)
        used += take_portions(state, item, amount - used, day, consumed_by,
                              dish_id)
    return used


def update_visibility(state, busy):
    """Each morning: which items the person overlooks today."""
    for item in state["fridge"]:
        probability = FORGET_PROBABILITY[item["category"]]
        probability *= PLACE_FORGET_FACTOR[item["place"]]
        probability *= PERSONAL_FORGET_FACTOR.get(item["name"], 1.0)
        if item["opened_date"] is not None:
            probability *= OPENED_FORGET_FACTOR
        if busy:
            probability *= BUSY_FORGET_FACTOR
        item["visible"] = random.random() >= probability


def mark_spoiled_items(state, day):
    """End of every day: items at their last good day become waste and get
    the day they will be thrown away."""
    for item in state["fridge"]:
        if item["spoiled"]:
            continue
        if last_good_day(item) <= day:
            item["spoiled"] = True
            if random.random() < P_NOTICE_LATE:
                delay = random.randint(LATE_NOTICE_DAYS[0], LATE_NOTICE_DAYS[1])
            else:
                delay = 1
            item["removal_day"] = day + datetime.timedelta(days=delay)


def throw_away_spoiled(state, day):
    """Morning at home: throw away spoiled items that are noticed by now."""
    for item in list(state["fridge"]):
        if item["spoiled"] and item["removal_day"] <= day:
            remove_item(state, item, day, "wasted", None, None)


# ===========================================================================
# EATING
# ===========================================================================

def eat_snack(state, day, template_weights):
    """One eating occasion without cooking. Returns True if something was
    eaten."""
    names = []
    weights = []
    for name in template_weights:
        first_slot_items = []
        for wanted in SNACK_TEMPLATES[name][0][0]:
            first_slot_items.extend(usable_items(state, day, wanted))
        if first_slot_items:
            # The most appealing food available for the template counts.
            best_appeal = 0.0
            for item in first_slot_items:
                if appeal(item["name"]) > best_appeal:
                    best_appeal = appeal(item["name"])
            names.append(name)
            weights.append(template_weights[name]
                           * preference_factor(first_slot_items, day)
                           * best_appeal)
    if not names:
        return False

    template = random.choices(names, weights)[0]
    for options, probability in SNACK_TEMPLATES[template]:
        if random.random() >= probability:
            continue
        candidates = []
        for wanted in options:
            candidates.extend(usable_items(state, day, wanted))
        if candidates:
            item = choose_item(candidates, day)
            take_portions(state, item, 1, day, "snack", None)
    return True


def maybe_eat_leftovers(state, day, base_probability):
    """Eat one portion of leftovers if there are some and they appeal.
    Returns True if leftovers were eaten."""
    candidates = usable_items(state, day, "leftovers")
    if not candidates:
        return False
    item = choose_item(candidates, day)
    age = (day - item["added_date"]).days
    probability = base_probability
    probability *= item["source_dish"]["rating"] / LEFTOVERS_RATING_REFERENCE
    probability *= LEFTOVERS_FATIGUE ** (age - 1)
    if random.random() >= probability:
        return False
    take_portions(state, item, 1, day, "dish", item["source_dish"]["id"])
    return True


# ===========================================================================
# COOKING
# ===========================================================================

def dish_preference(state, day, dish):
    """Weight of a dish from rating, season and variety."""
    weight = dish["rating"]
    if dish["season"] == "winter":
        if day.month in WINTER_MONTHS:
            weight *= DISH_IN_SEASON_FACTOR
        else:
            weight *= DISH_OUT_OF_SEASON_FACTOR
    elif dish["season"] == "summer":
        if day.month in SUMMER_MONTHS:
            weight *= DISH_IN_SEASON_FACTOR
        else:
            weight *= DISH_OUT_OF_SEASON_FACTOR
    last_cooked = state["last_cooked"].get(dish["id"])
    if last_cooked is not None:
        if (day - last_cooked).days < RECENTLY_COOKED_DAYS:
            weight *= RECENTLY_COOKED_FACTOR
    return weight


def can_cook(state, day, dish):
    key_name, key_amount = dish["ingredients"][0]
    if total_portions(usable_items(state, day, key_name)) < key_amount:
        return False
    if key_ingredient_expiring(state, day, dish):
        # Improvising to use up the key ingredient before it spoils.
        return True
    others = dish["ingredients"][1:]
    available = 0
    for name, amount in others:
        if usable_items(state, day, name):
            available += 1
    return available >= MIN_OTHER_INGREDIENTS_SHARE * len(others)


def key_ingredient_expiring(state, day, dish):
    """True if the key ingredient is at most USE_UP_DAYS from spoiling."""
    key_name = dish["ingredients"][0][0]
    for item in usable_items(state, day, key_name):
        if (last_good_day(item) - day).days <= USE_UP_DAYS:
            return True
    return False


def choose_dish_to_cook(state, day):
    """A dish that can be cooked with what is in the fridge, or None."""
    options = []
    weights = []
    for dish in state["dishes"]:
        if not can_cook(state, day, dish):
            continue
        weight = dish_preference(state, day, dish)
        if key_ingredient_expiring(state, day, dish):
            weight *= USE_UP_FACTOR
        options.append(dish)
        weights.append(weight)
    if not options:
        return None
    return random.choices(options, weights)[0]


def cook_dish(state, day, dish):
    for name, amount in dish["ingredients"]:
        use_portions(state, day, name, amount, "dish", dish["id"])
    if random.random() < P_SIDE_VEGETABLE:
        candidates = usable_items(state, day, "vegetable")
        if candidates:
            item = choose_item(candidates, day)
            take_portions(state, item, 1, day, "dish", dish["id"])
    state["cooking_log"].append((day, dish["id"]))
    state["last_cooked"][dish["id"]] = day

    portions = dish["servings"] - 1 + random.randint(
        -LEFTOVER_PORTIONS_NOISE, LEFTOVER_PORTIONS_NOISE)
    if portions > 0:
        add_leftovers(state, day, dish, portions)


def cook_probability(day):
    probability = COOK_PROBABILITY[day.weekday()]
    if day.month in SUMMER_MONTHS:
        probability *= SUMMER_COOK_FACTOR
    if in_christmas_period(day):
        probability *= CHRISTMAS_COOK_FACTOR
    return probability


def eat_out_probability(day):
    probability = EAT_OUT_PROBABILITY[day.weekday()]
    if day.month in SUMMER_MONTHS:
        probability *= SUMMER_EAT_OUT_FACTOR
    return probability


def eat_dinner(state, day, busy):
    if busy:
        if random.random() < P_EAT_OUT_BUSY:
            return
        if maybe_eat_leftovers(state, day, P_LEFTOVERS_DINNER):
            return
        eat_snack(state, day, SNACK_DINNER_TEMPLATES)
        return

    # The dish the person would cook, if any. If its key ingredient is about
    # to spoil, cooking comes before leftovers and is more likely.
    dish = choose_dish_to_cook(state, day)
    using_up = dish is not None and key_ingredient_expiring(state, day, dish)
    if not using_up and maybe_eat_leftovers(state, day, P_LEFTOVERS_DINNER):
        return

    probability = cook_probability(day)
    if using_up and probability < P_COOK_WHEN_USING_UP:
        probability = P_COOK_WHEN_USING_UP
    if random.random() < probability:
        if dish is None and random.random() < P_TOPUP_FOR_DISH:
            dish = shop_for_dish(state, day)
        if dish is not None:
            cook_dish(state, day, dish)
            return

    if random.random() < eat_out_probability(day):
        return
    eat_snack(state, day, SNACK_DINNER_TEMPLATES)


# ===========================================================================
# SHOPPING
# ===========================================================================

def packages_needed(product, portions):
    return (portions + product["portions"] - 1) // product["portions"]


def known_items(state, checked_fridge):
    """What the person knows is at home when shopping: what they see when
    checking the fridge, otherwise what they happen to remember."""
    known = []
    for item in state["fridge"]:
        if item["spoiled"] or not item["visible"]:
            continue
        if checked_fridge or random.random() < P_REMEMBER_ITEM:
            known.append(item)
    return known


def known_portions(known, name):
    total = 0
    for item in known:
        if item["name"] == name:
            total += item["portions_left"]
    return total


def known_packages(known, product):
    """Packages of the product known to be at home with enough left."""
    count = 0
    for item in known:
        if item["name"] != product["name"]:
            continue
        if item["portions_left"] >= ENOUGH_LEFT_SHARE * product["portions"]:
            count += 1
    return count


def plan_dishes(state, day, count):
    """Dishes for the coming week: different key ingredients, key ingredient
    in season, chosen by preference."""
    plan = []
    used_keys = []
    for _ in range(count):
        options = []
        weights = []
        for dish in state["dishes"]:
            key_name = dish["ingredients"][0][0]
            if dish in plan or key_name in used_keys:
                continue
            if not in_season(state["catalogue"][key_name], day):
                continue
            options.append(dish)
            weights.append(dish_preference(state, day, dish))
        if not options:
            break
        dish = random.choices(options, weights)[0]
        plan.append(dish)
        used_keys.append(dish["ingredients"][0][0])
    return plan


def impulse_buys(state, day):
    """Overbuying that can happen on any shopping trip."""
    if random.random() < P_SHOPPING_HUNGRY:
        options = []
        for name in IMPULSE_PRODUCTS:
            if in_season(state["catalogue"][name], day):
                options.append(state["catalogue"][name])
        count = random.randint(HUNGRY_EXTRA_ITEMS[0], HUNGRY_EXTRA_ITEMS[1])
        for _ in range(count):
            buy(state, day, random.choice(options), 1)

    if random.random() < P_RED_LABEL:
        name = random.choice(RED_LABEL_PRODUCTS)
        buy(state, day, state["catalogue"][name], 1, red_label=True)


def weekly_shop(state, day):
    catalogue = state["catalogue"]
    checked_fridge = random.random() < P_CHECK_FRIDGE
    known = known_items(state, checked_fridge)

    days_to_trip, trip_length = next_trip(day)
    remembers_trip = False
    if days_to_trip is not None and days_to_trip <= TRIP_PLANNING_DAYS:
        if trip_length >= LONG_TRIP_DAYS:
            remembers_trip = True
        else:
            remembers_trip = random.random() < P_REMEMBER_TRIP

    # 1. Ingredients for the dishes planned for the coming week.
    dish_count = random.randint(PLANNED_DISHES_PER_WEEK[0],
                                PLANNED_DISHES_PER_WEEK[1])
    if remembers_trip:
        dish_count = min(dish_count, days_to_trip // TRIP_DAYS_PER_DISH)
    needed = {}
    for dish in plan_dishes(state, day, dish_count):
        for name, amount in dish["ingredients"]:
            needed[name] = needed.get(name, 0) + amount
    for name in needed:
        product = catalogue[name]
        if not in_season(product, day):
            continue
        missing = needed[name] - known_portions(known, name)
        if missing > 0:
            known.extend(buy(state, day, product,
                             packages_needed(product, missing)))

    # 2. Restocking regularly bought products.
    skip_restock = remembers_trip and days_to_trip <= TRIP_NO_RESTOCK_DAYS
    if not skip_restock:
        for product in catalogue.values():
            if product["restock"] == 0 or not in_season(product, day):
                continue
            habit = product["name"] in HABIT_PRODUCT_APPEAL
            if known_packages(known, product) > 0 and not habit:
                continue
            if random.random() >= product["restock"]:
                continue
            packages = 1
            if product["name"] in CHRISTMAS_FOODS and in_christmas_period(day):
                if random.random() < P_CHRISTMAS_EXTRA_PACKAGE:
                    packages += 1
            known.extend(buy(state, day, product, packages))

    # 3. Special offer.
    if random.random() < P_SPECIAL_OFFER:
        options = []
        for product in catalogue.values():
            if product["restock"] > 0 and in_season(product, day):
                options.append(product)
        count = random.randint(OFFER_EXTRA_PACKAGES[0], OFFER_EXTRA_PACKAGES[1])
        buy(state, day, random.choice(options), count)

    # 4. Hungry shopping and red-label discounts.
    impulse_buys(state, day)


def maybe_top_up(state, day):
    """Small shopping trip when an essential has run out (not just before
    leaving on a trip)."""
    days_to_trip, trip_length = next_trip(day)
    if days_to_trip is not None and days_to_trip <= TRIP_NO_RESTOCK_DAYS:
        return
    missing = []
    for wanted, product_name in TOPUP_ESSENTIALS.items():
        if not usable_items(state, day, wanted):
            missing.append(product_name)
    if not missing or random.random() >= P_TOPUP_WHEN_OUT:
        return
    for product_name in missing:
        buy(state, day, state["catalogue"][product_name], 1)
    impulse_buys(state, day)


def shop_for_dish(state, day):
    """Quick trip to buy what is missing for one dish. Returns the dish."""
    options = []
    weights = []
    for dish in state["dishes"]:
        key_product = state["catalogue"][dish["ingredients"][0][0]]
        if in_season(key_product, day):
            options.append(dish)
            weights.append(dish_preference(state, day, dish))
    dish = random.choices(options, weights)[0]

    for name, amount in dish["ingredients"]:
        product = state["catalogue"][name]
        if not in_season(product, day):
            continue
        missing = amount - total_portions(usable_items(state, day, name))
        if missing > 0:
            buy(state, day, product, packages_needed(product, missing))
    impulse_buys(state, day)
    return dish


# ===========================================================================
# SIMULATION
# ===========================================================================

def live_one_day(state, day, big_shop_today):
    busy = in_periods(day, BUSY_PERIODS)
    throw_away_spoiled(state, day)
    update_visibility(state, busy)
    weekend = day.weekday() >= 5

    snack_factor = 1.0
    if busy:
        snack_factor = BUSY_HOME_SNACK_FACTOR

    # Morning.
    if random.random() < P_COFFEE_MILK:
        use_portions(state, day, "Milk", 1, "snack", None)
    if random.random() < P_BREAKFAST * snack_factor:
        eat_snack(state, day, BREAKFAST_TEMPLATES)

    # Lunch: at home on weekends, at work on weekdays (maybe a lunch box).
    if weekend:
        if not maybe_eat_leftovers(state, day, P_LEFTOVERS_LUNCH):
            eat_snack(state, day, WEEKEND_LUNCH_TEMPLATES)
    else:
        maybe_eat_leftovers(state, day, P_LUNCHBOX)

    if random.random() < P_AFTERNOON_SNACK * snack_factor:
        eat_snack(state, day, AFTERNOON_SNACK_TEMPLATES)

    # Shopping before dinner.
    if big_shop_today:
        weekly_shop(state, day)
    else:
        maybe_top_up(state, day)

    eat_dinner(state, day, busy)

    # Late evening.
    probability = P_LATE_SNACK_WEEKDAY
    if day.weekday() in WEEKEND_EVENINGS:
        probability = P_LATE_SNACK_WEEKEND
    if random.random() < probability:
        eat_snack(state, day, LATE_SNACK_TEMPLATES)


def simulate(state):
    day = START_DATE
    big_shop_day = START_DATE
    while day <= END_DATE:
        if day.weekday() == 0 and day != START_DATE:
            big_shop_day = choose_big_shop_day(day)
        if not in_periods(day, TRAVEL_PERIODS):
            live_one_day(state, day, big_shop_day == day)
        mark_spoiled_items(state, day)
        day += ONE_DAY


# ===========================================================================
# OUTPUT RECORDS AND CONSISTENCY CHECKS
# ===========================================================================

def date_text(day):
    if day is None:
        return None
    return day.isoformat()


def build_history(state):
    """Records of all items removed during the period. Items still in the
    fridge on the last day have no outcome yet and are left out."""
    records = []
    still_in_fridge = 0
    for item in state["items"]:
        if item["removed_date"] is None:
            still_in_fridge += 1
            continue
        records.append({
            "id": len(records) + 1,
            "name": item["name"],
            "category": item["category"],
            "barcode": item["barcode"],
            "added_date": date_text(item["added_date"]),
            "expiry_date": date_text(item["expiry_date"]),
            "opened_date": date_text(item["opened_date"]),
            "removed_date": date_text(item["removed_date"]),
            "outcome": item["outcome"],
            "consumed_by": item["consumed_by"],
            "dish_id": item["dish_id"],
        })
    return records, still_in_fridge


def build_dish_records(state):
    records = []
    for dish in state["dishes"]:
        times_cooked = 0
        for day, dish_id in state["cooking_log"]:
            if dish_id == dish["id"]:
                times_cooked += 1
        main_ingredients = []
        for name, amount in dish["ingredients"]:
            main_ingredients.append(name)
        records.append({
            "id": dish["id"],
            "name": dish["name"],
            "main_ingredients": main_ingredients,
            "rating": dish["rating"],
            "times_cooked": times_cooked,
        })
    return records


def build_catalogue_records(state):
    """Product data a real app would have in its product database."""
    records = []
    for product in state["catalogue"].values():
        records.append({
            "name": product["name"],
            "category": product["category"],
            "shelf_life_days": product["shelf_life"],
            "opened_shelf_life_days": product["opened_life"],
        })
    return records


def cooking_days_in_history(history):
    """{dish_id: set of days} read from the history: a dish was cooked on a
    day when a (non-leftovers) item was used up in it on that day."""
    days = {}
    for record in history:
        if record["consumed_by"] != "dish":
            continue
        if record["category"] == "leftovers":
            continue
        if record["dish_id"] not in days:
            days[record["dish_id"]] = set()
        days[record["dish_id"]].add(record["removed_date"])
    return days


def check_consistency(history, dish_records, catalogue_records):
    errors = []
    dish_ids = set()
    for dish in dish_records:
        dish_ids.add(dish["id"])
    catalogue_categories = {}
    for product in catalogue_records:
        catalogue_categories[product["name"]] = product["category"]
    names = set()
    for record in history:
        names.add(record["name"])
        names.add(record["category"])

    for record in history:
        label = "item " + str(record["id"]) + " (" + record["name"] + ")"
        added = datetime.date.fromisoformat(record["added_date"])
        expiry = datetime.date.fromisoformat(record["expiry_date"])
        removed = datetime.date.fromisoformat(record["removed_date"])

        if not START_DATE <= added <= END_DATE:
            errors.append(label + ": added outside the period")
        if expiry < added:
            errors.append(label + ": expiry before added")
        if record["opened_date"] is None:
            if not added <= removed:
                errors.append(label + ": removed before added")
        else:
            opened = datetime.date.fromisoformat(record["opened_date"])
            if not added <= opened <= removed:
                errors.append(label + ": dates not added <= opened <= removed")

        if record["outcome"] == "eaten":
            if removed > expiry:
                errors.append(label + ": eaten after its expiry date")
            if record["consumed_by"] not in ("dish", "snack"):
                errors.append(label + ": eaten without consumed_by")
        elif record["outcome"] == "wasted":
            if record["consumed_by"] is not None:
                errors.append(label + ": wasted but consumed_by is set")
        else:
            errors.append(label + ": unknown outcome")

        if record["consumed_by"] == "dish":
            if record["dish_id"] not in dish_ids:
                errors.append(label + ": dish_id is not an existing dish")
        elif record["dish_id"] is not None:
            errors.append(label + ": dish_id set but not eaten in a dish")

        if record["category"] != "leftovers":
            if catalogue_categories.get(record["name"]) != record["category"]:
                errors.append(label + ": not in the catalogue with this category")

    cooking_days = cooking_days_in_history(history)
    for dish in dish_records:
        days = cooking_days.get(dish["id"], set())
        if len(days) != dish["times_cooked"]:
            errors.append(dish["name"] + ": times_cooked does not match history")
        for name in dish["main_ingredients"]:
            if name not in names:
                errors.append(dish["name"] + ": ingredient " + name
                              + " not used in the history")

    if errors:
        raise ValueError("Consistency check failed:\n" + "\n".join(errors))


# ===========================================================================
# DATA_README.md
# ===========================================================================

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
               "Sep", "Oct", "Nov", "Dec"]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]


def pct(value):
    """0.35 -> '35 %'"""
    return f"{value * 100:g} %"


def rate_text(part, whole):
    if whole == 0:
        return "-"
    return f"{part / whole * 100:.1f} %"


def month_day_text(month_day):
    month, day = month_day
    return str(day) + " " + MONTH_NAMES[month - 1]


def season_text(season):
    if season is None:
        return "all year"
    parts = []
    for start, end in season:
        parts.append(month_day_text(start) + "-" + month_day_text(end))
    return ", ".join(parts)


def months_text(months):
    parts = []
    for month in months:
        parts.append(MONTH_NAMES[month - 1])
    return ", ".join(parts)


def range_text(low_high):
    return str(low_high[0]) + "-" + str(low_high[1])


def summarize(history, dish_records):
    added_dates = []
    removed_dates = []
    categories = {}
    products = {}
    wasted = 0
    eaten_ingredients = 0
    eaten_leftovers = 0
    eaten_snacks = 0
    for record in history:
        added_dates.append(record["added_date"])
        removed_dates.append(record["removed_date"])
        category = record["category"]
        if category not in categories:
            categories[category] = {"items": 0, "wasted": 0}
        categories[category]["items"] += 1
        if category != "leftovers":
            if record["name"] not in products:
                products[record["name"]] = {"items": 0, "wasted": 0}
            products[record["name"]]["items"] += 1

        if record["outcome"] == "wasted":
            wasted += 1
            categories[category]["wasted"] += 1
            if category != "leftovers":
                products[record["name"]]["wasted"] += 1
        elif record["consumed_by"] == "snack":
            eaten_snacks += 1
        elif category == "leftovers":
            eaten_leftovers += 1
        else:
            eaten_ingredients += 1

    cookings = 0
    for dish in dish_records:
        cookings += dish["times_cooked"]

    return {
        "items": len(history),
        "wasted": wasted,
        "first_added": min(added_dates),
        "last_added": max(added_dates),
        "first_removed": min(removed_dates),
        "last_removed": max(removed_dates),
        "categories": categories,
        "products": products,
        "eaten_ingredients": eaten_ingredients,
        "eaten_leftovers": eaten_leftovers,
        "eaten_snacks": eaten_snacks,
        "cookings": cookings,
    }


def template_text(slots):
    parts = []
    for options, probability in slots:
        if len(options) == 1:
            text = options[0]
        else:
            text = "one of " + " / ".join(options)
        if probability >= 1.0:
            text += " (always)"
        else:
            text += " (" + pct(probability) + ")"
        parts.append(text)
    return ", ".join(parts)


def weights_text(weights):
    parts = []
    for name in weights:
        parts.append(name + " " + str(weights[name]))
    return ", ".join(parts)


def write_readme(history, dish_records, state, still_in_fridge, real_barcodes):
    stats = summarize(history, dish_records)
    eaten = stats["items"] - stats["wasted"]
    lines = []

    def add(text=""):
        lines.append(text)

    add("All data in this dataset is simulated. It was generated by a Python "
        "script written by " + MODEL_NAME + " (AI). The data is cleaner than real app data would be: it has no missed scans, wrong dates or unmarked items.")
    add()
    add("# Simulated fridge usage history")
    add()
    add("## Dataset")
    add()
    add("**Purpose.** Training and testing a scikit-learn classifier for a "
        "fridge management app that predicts whether a fridge item will be "
        "eaten or wasted.")
    add()
    add("**Simulated household.** One adult living alone in Finland. Works "
        "on weekdays and eats lunch at work, cooks favourite dishes at home a "
        "few times a week, snacks during the day and sometimes late in the "
        "evening, and keeps leftovers in the fridge.")
    add()
    add("**Period.** " + START_DATE.isoformat() + " to " + END_DATE.isoformat()
        + " (12 months), simulated day by day.")
    add()
    add("**Files.**")
    add()
    add("- `" + HISTORY_FILE + "`: one record per fridge item, from the moment "
        "it is added until it is removed (JSON array).")
    add("- `" + DISHES_FILE + "`: the favourite dishes (JSON array).")
    add("- `" + CATALOGUE_FILE + "`: the product catalogue with shelf lives "
        "(JSON array), as a real app would have in its product database.")
    add("- `" + README_FILE + "`: this file, written by `" + SCRIPT_NAME
        + "` together with the data.")
    add()
    add("### Fields of " + HISTORY_FILE)
    add()
    add("| Field | Type | Meaning |")
    add("|---|---|---|")
    add("| id | int | Unique item id, 1, 2, ... in the order the items were "
        "added. |")
    add("| name | str | Product name, e.g. \"Cucumber\". Leftovers are named "
        "\"<dish name> (leftovers)\". |")
    add("| category | str | dairy, eggs, meat, fish, bread, vegetable, fruit, "
        "ready_meal, condiment or leftovers. |")
    add("| barcode | str | EAN-13 code. Real code from `" + PRODUCTS_FILE
        + "` if the product is listed there, otherwise a fake store-internal "
        "code (prefix 20-29, valid check digit). Every product has one code; "
        "the leftovers of each dish have their own fake code. |")
    add("| added_date | str (YYYY-MM-DD) | Day the item was put in the fridge "
        "(bought, or cooked for leftovers). |")
    add("| expiry_date | str (YYYY-MM-DD) | Date on the package; for leftovers "
        "the cooking day + " + str(LEFTOVERS_SHELF_LIFE) + " days. |")
    add("| opened_date | str (YYYY-MM-DD) or null | Day the first portion was "
        "used. null if the item was never opened. |")
    add("| removed_date | str (YYYY-MM-DD) | Day the item left the fridge: "
        "last portion eaten, or thrown away. |")
    add("| outcome | str | \"eaten\" (all portions eaten) or \"wasted\" "
        "(thrown away, even if partly eaten). |")
    add("| consumed_by | str or null | How the last portion was eaten: "
        "\"dish\" = used in cooking a favourite dish, or leftovers of a dish "
        "eaten later; \"snack\" = eaten without cooking (breakfast, sandwich, "
        "snack, late-evening snack, ready meal); null if wasted. |")
    add("| dish_id | int or null | Id of the favourite dish the item was used "
        "in (for leftovers: the dish they came from). null for snacks and "
        "wasted items. |")
    add()
    add("What a real app could know, and may be used as model features:")
    add()
    add("- When an item is added: name, category, barcode, added_date, "
        "expiry_date, and the product's shelf lives from `" + CATALOGUE_FILE
        + "`.")
    add("- On a later day: whether the item has been opened (opened_date "
        "before that day), which items are in the fridge that day, and the "
        "outcomes of items removed before that day (e.g. each product's past "
        "waste rate).")
    add("- Not as features: removed_date, outcome, consumed_by and dish_id "
        "of the item itself, which are only known when it leaves the fridge "
        "and reveal the label.")
    add("- Not in the data at all: the personal habits (where products are "
        "kept, blind spots, habit purchases; see Simulation assumptions). An "
        "app cannot know them directly; it can only learn their effect from "
        "the history, e.g. as a product's past waste rate. No new fields were "
        "added to " + HISTORY_FILE + " for them.")
    add()
    add("### Fields of " + DISHES_FILE)
    add()
    add("| Field | Type | Meaning |")
    add("|---|---|---|")
    add("| id | int | Dish id, referenced by dish_id in " + HISTORY_FILE + ". |")
    add("| name | str | Dish name. |")
    add("| main_ingredients | list of str | Product names as used in the name "
        "field of " + HISTORY_FILE + ". The first one is the key ingredient. |")
    add("| rating | int | How much the person likes the dish, 1-5. |")
    add("| times_cooked | int | Times cooked during the period. Equals the "
        "number of different removed_date values among non-leftovers records "
        "with this dish_id and consumed_by \"dish\". |")
    add()
    add("### Fields of " + CATALOGUE_FILE)
    add()
    add("| Field | Type | Meaning |")
    add("|---|---|---|")
    add("| name | str | Product name as used in " + HISTORY_FILE + ". |")
    add("| category | str | Product category. |")
    add("| shelf_life_days | int | Typical days from purchase to the date on "
        "the package. The real expiry_date of an item varies (see Noise). |")
    add("| opened_shelf_life_days | int | Typical days the product keeps after "
        "opening; an opened item is good until opened_date + this, but not "
        "after its expiry_date. |")
    add()
    add("Leftovers are not in the catalogue; their expiry_date already is the "
        "end of their shelf life. All catalogue fields are information a real "
        "app has, so they may be used as model features.")
    add()

    add("## Summary statistics")
    add()
    add("- Items: " + str(stats["items"]))
    add("- Added: " + stats["first_added"] + " to " + stats["last_added"]
        + "; removed: " + stats["first_removed"] + " to "
        + stats["last_removed"])
    add("- Overall waste rate: " + rate_text(stats["wasted"], stats["items"])
        + " (" + str(stats["wasted"]) + " wasted, " + str(eaten) + " eaten)")
    add("- Eaten items by consumption type: dish "
        + str(stats["eaten_ingredients"] + stats["eaten_leftovers"])
        + " (" + str(stats["eaten_ingredients"]) + " ingredients, "
        + str(stats["eaten_leftovers"]) + " leftovers), snack "
        + str(stats["eaten_snacks"]))
    add("- Dishes cooked: " + str(stats["cookings"]) + " times")
    add("- Not included: " + str(still_in_fridge) + " items that were still in "
        "the fridge on " + END_DATE.isoformat() + " (no outcome yet)")
    add("- Real barcodes from " + PRODUCTS_FILE + ": " + str(real_barcodes)
        + " of " + str(len(state["catalogue"])) + " products")
    add()
    add("Waste rate per category:")
    add()
    add("| Category | Items | Wasted | Waste rate |")
    add("|---|---|---|---|")
    category_rows = []
    for category in stats["categories"]:
        counts = stats["categories"][category]
        category_rows.append((counts["wasted"] / counts["items"], category))
    category_rows.sort(reverse=True)
    for rate, category in category_rows:
        counts = stats["categories"][category]
        add("| " + category + " | " + str(counts["items"]) + " | "
            + str(counts["wasted"]) + " | "
            + rate_text(counts["wasted"], counts["items"]) + " |")
    add()
    add("Waste rate per product (leftovers not included):")
    add()
    add("| Product | Items | Wasted | Waste rate |")
    add("|---|---|---|---|")
    product_rows = []
    for name in stats["products"]:
        counts = stats["products"][name]
        product_rows.append((counts["wasted"] / counts["items"], name))
    product_rows.sort(reverse=True)
    for rate, name in product_rows:
        counts = stats["products"][name]
        add("| " + name + " | " + str(counts["items"]) + " | "
            + str(counts["wasted"]) + " | "
            + rate_text(counts["wasted"], counts["items"]) + " |")
    add()
    add("Times each dish was cooked:")
    add()
    add("| Id | Dish | Rating | Times cooked |")
    add("|---|---|---|---|")
    for dish in dish_records:
        add("| " + str(dish["id"]) + " | " + dish["name"] + " | "
            + str(dish["rating"]) + " | " + str(dish["times_cooked"]) + " |")
    add()

    add("## Simulation assumptions")
    add()
    add("All values below are the constants at the top of `" + SCRIPT_NAME
        + "`. Probabilities are per occasion unless stated otherwise.")
    add()
    add("### Daily routine")
    add()
    add("- The simulation starts with an empty fridge and a big shop on the "
        "first day.")
    add("- A day at home, in order: spoiled items that have been noticed are "
        "thrown away; items that are overlooked today are chosen (see Waste "
        "tendencies); coffee with milk (" + pct(P_COFFEE_MILK) + ", one "
        "portion of milk); breakfast (" + pct(P_BREAKFAST) + "); lunch; "
        "afternoon snack (" + pct(P_AFTERNOON_SNACK) + "); shopping; dinner; "
        "late-evening snack.")
    add("- Lunch: Monday-Friday at work, with leftovers taken in a lunch box "
        "(see Leftovers). Saturday-Sunday at home: leftovers, otherwise a "
        "snack.")
    add("- Dinner on a normal day: first the dish the person would cook is "
        "chosen (see Cooking). If its key ingredient is at most "
        + str(USE_UP_DAYS) + " days from its last good day, leftovers are "
        "skipped and the person cooks with at least "
        + pct(P_COOK_WHEN_USING_UP) + ". Otherwise leftovers come first if "
        "there are some and they appeal (see Leftovers), then cooking with "
        "the weekday probability below. If no dish can be made, the person "
        "shops for one dish with " + pct(P_TOPUP_FOR_DISH) + ". Not cooking: "
        "eating out with the weekday probability below, otherwise a snack "
        "dinner.")
    weekend_evenings = []
    for weekday in WEEKEND_EVENINGS:
        weekend_evenings.append(WEEKDAY_NAMES[weekday])
    add("- Late-evening snack: " + pct(P_LATE_SNACK_WEEKEND) + " on "
        + " and ".join(weekend_evenings) + " evenings, "
        + pct(P_LATE_SNACK_WEEKDAY) + " on other evenings.")
    add("- At the end of every day (also when away) items at their last good "
        "day become spoiled.")
    add()
    add("| Weekday | Cook | Eat out |")
    add("|---|---|---|")
    for index in range(7):
        add("| " + WEEKDAY_NAMES[index] + " | " + pct(COOK_PROBABILITY[index])
            + " | " + pct(EAT_OUT_PROBABILITY[index]) + " |")
    add()

    add("### Shopping")
    add()
    shop_days = []
    for weekday in BIG_SHOP_DAY_WEIGHTS:
        shop_days.append(WEEKDAY_NAMES[weekday] + " "
                         + pct(BIG_SHOP_DAY_WEIGHTS[weekday]))
    add("- One bigger weekly shop; its day is chosen every Monday: "
        + ", ".join(shop_days) + ". Skipped if the person is away that day.")
    add("- Before the weekly shop the person checks the fridge with "
        + pct(P_CHECK_FRIDGE) + ". Otherwise each item at home is remembered "
        "with " + pct(P_REMEMBER_ITEM) + ", so products already at home are "
        "bought again. Overlooked and spoiled items are never counted.")
    add("- Planned dishes: " + range_text(PLANNED_DISHES_PER_WEEK)
        + " favourite dishes for the coming week, chosen by preference (see "
        "Cooking), with different key ingredients and the key ingredient in "
        "season. Missing portions of their ingredients are bought as whole "
        "packages; out-of-season ingredients are skipped.")
    habit_names = []
    for name in HABIT_PRODUCT_APPEAL:
        habit_names.append(name)
    add("- Restocking: every product in season with a restock probability "
        "above 0 is bought (one package) with that probability when no "
        "package of it with at least " + pct(ENOUGH_LEFT_SHARE)
        + " of the portions left is known to be at home. Habit products ("
        + ", ".join(habit_names) + ") are bought with their restock "
        "probability at every weekly shop, even when some is at home.")
    add("- Trips: if a trip starts within " + str(TRIP_PLANNING_DAYS)
        + " days, the person remembers it at the weekly shop with "
        + pct(P_REMEMBER_TRIP) + "; a holiday of at least "
        + str(LONG_TRIP_DAYS) + " days is always remembered. Then at most one "
        "dish per " + str(TRIP_DAYS_PER_DISH) + " days before the trip is "
        "planned, and nothing is restocked if the trip starts within "
        + str(TRIP_NO_RESTOCK_DAYS) + " days. There are no top-up trips in "
        "those days either.")
    essentials = []
    for wanted in TOPUP_ESSENTIALS:
        essentials.append("no " + wanted + " -> buy " + TOPUP_ESSENTIALS[wanted])
    add("- Top-up trips on other days: when an essential is not found in the "
        "fridge (" + ", ".join(essentials) + "), the person goes to the shop "
        "with " + pct(P_TOPUP_WHEN_OUT) + ". Shopping for one dish: see "
        "Daily routine.")
    add("- Special offers: at a weekly shop, with " + pct(P_SPECIAL_OFFER)
        + " a random regularly bought product is on offer and "
        + range_text(OFFER_EXTRA_PACKAGES) + " extra packages are bought.")
    add("- Shopping while hungry: on any trip, with " + pct(P_SHOPPING_HUNGRY)
        + " " + range_text(HUNGRY_EXTRA_ITEMS) + " impulse items are bought "
        "from: " + ", ".join(IMPULSE_PRODUCTS) + " (if in season).")
    add("- Red-label discounts: on any trip, with " + pct(P_RED_LABEL)
        + " one discounted item with only " + range_text(RED_LABEL_DAYS_LEFT)
        + " days left is bought from: " + ", ".join(RED_LABEL_PRODUCTS) + ".")
    add()

    add("### Products and shelf lives")
    add()
    add("Expiry date = purchase day + shelf life (+- noise). After opening, "
        "the item is good for the opened shelf life (+- noise), but never "
        "longer than the expiry date. A portion is one use (one coffee milk, "
        "one sandwich topping, one yogurt bowl, ...).")
    add()
    add("| Product | Category | Shelf life (days) | Opened (days) | Portions "
        "| Restock | Place | Season |")
    add("|---|---|---|---|---|---|---|---|")
    for product in state["catalogue"].values():
        add("| " + product["name"] + " | " + product["category"] + " | "
            + str(product["shelf_life"]) + " | " + str(product["opened_life"])
            + " | " + str(product["portions"]) + " | " + pct(product["restock"])
            + " | " + product["place"] + " | "
            + season_text(product["season"]) + " |")
    add()

    add("### Cooking and favourite dishes")
    add()
    add("- A dish can be cooked when its key ingredient (first, in bold) is in "
        "the fridge in full and at least "
        + pct(MIN_OTHER_INGREDIENTS_SHARE) + " of the other ingredients are "
        "available; missing ones are left out. If the key ingredient is at "
        "most " + str(USE_UP_DAYS) + " days from its last good day, the "
        "person improvises and the other ingredients are not needed. The key "
        "ingredient amount is one whole package, so every cooking uses up at "
        "least one item.")
    add("- The dish is chosen among the possible ones with weight = rating x "
        + str(DISH_IN_SEASON_FACTOR) + " in its season (or x "
        + str(DISH_OUT_OF_SEASON_FACTOR) + " outside it) x "
        + str(RECENTLY_COOKED_FACTOR) + " if cooked during the last "
        + str(RECENTLY_COOKED_DAYS) + " days x " + str(USE_UP_FACTOR)
        + " if its key ingredient is at most " + str(USE_UP_DAYS) + " days "
        "from its last good day. Planning and shopping for a dish use the "
        "same weights without the last factor.")
    add("- With " + pct(P_SIDE_VEGETABLE) + " one portion of a vegetable "
        "(chosen like a snack item) is eaten with the dish as a side.")
    add("- Winter months: " + months_text(WINTER_MONTHS) + ". Summer months: "
        + months_text(SUMMER_MONTHS) + ".")
    add()
    add("| Dish | Rating | Season | Servings | Ingredients (portions) |")
    add("|---|---|---|---|---|")
    for dish in state["dishes"]:
        parts = []
        for index in range(len(dish["ingredients"])):
            name, amount = dish["ingredients"][index]
            if index == 0:
                parts.append("**" + name + "** " + str(amount))
            else:
                parts.append(name + " " + str(amount))
        add("| " + dish["name"] + " | " + str(dish["rating"]) + " | "
            + dish["season"] + " | " + str(dish["servings"]) + " | "
            + ", ".join(parts) + " |")
    add()

    add("### Snacking")
    add()
    add("Eating without cooking follows templates. A template is possible "
        "when its first part is in the fridge; each used part takes one "
        "portion of one item. A template's weight is multiplied by "
        + str(OPENED_PREFERENCE_FACTOR) + " if an item for its first part is "
        "opened and by " + str(USE_UP_FACTOR) + " if one is at most "
        + str(USE_UP_DAYS) + " days from its last good day, and by the "
        "highest appeal of the foods for its first part (see Personal "
        "habits). Within a part, the product is chosen with the same factors "
        "(base weight 1).")
    add()
    for name in SNACK_TEMPLATES:
        add("- " + name + ": " + template_text(SNACK_TEMPLATES[name]))
    add()
    add("Template weights per occasion (only possible templates count):")
    add()
    add("- Breakfast: " + weights_text(BREAKFAST_TEMPLATES))
    add("- Weekend lunch: " + weights_text(WEEKEND_LUNCH_TEMPLATES))
    add("- Afternoon snack: " + weights_text(AFTERNOON_SNACK_TEMPLATES))
    add("- Snack dinner: " + weights_text(SNACK_DINNER_TEMPLATES))
    add("- Late-evening snack: " + weights_text(LATE_SNACK_TEMPLATES))
    add()

    add("### Leftovers")
    add()
    add("- After cooking, one serving is eaten and servings - 1 (+- "
        + str(LEFTOVER_PORTIONS_NOISE) + ") portions go into one leftovers box "
        "(category leftovers). The box can be eaten from the next day and "
        "spoils after " + str(LEFTOVERS_SHELF_LIFE) + " days.")
    add("- Base probability of eating one portion when leftovers are found: "
        "weekday lunch box " + pct(P_LUNCHBOX) + ", weekend lunch "
        + pct(P_LEFTOVERS_LUNCH) + ", dinner " + pct(P_LEFTOVERS_DINNER)
        + ". It is multiplied by rating / " + str(LEFTOVERS_RATING_REFERENCE)
        + " and by " + str(LEFTOVERS_FATIGUE) + " for every day after the "
        "first. Leftovers are also a late-evening snack template.")
    add("- Eaten leftovers have consumed_by \"dish\" and the id of their dish.")
    add()

    add("### Travel and busy weeks")
    add()
    add("Away from home (nothing eaten or bought; spoiled items are thrown "
        "away after coming back):")
    add()
    add("| From | To | Reason |")
    add("|---|---|---|")
    for first_day, last_day, reason in TRAVEL_PERIODS:
        add("| " + first_day.isoformat() + " | " + last_day.isoformat() + " | "
            + reason + " |")
    add()
    add("Busy periods at home (no cooking; dinner eaten out with "
        + pct(P_EAT_OUT_BUSY) + ", otherwise leftovers or a snack dinner; "
        "breakfast and afternoon snack probabilities x"
        + str(BUSY_HOME_SNACK_FACTOR) + "; items overlooked x"
        + str(BUSY_FORGET_FACTOR) + " as often). They are not expected when "
        "shopping, so food bought for the week spoils:")
    add()
    add("| From | To | Reason |")
    add("|---|---|---|")
    for first_day, last_day, reason in BUSY_PERIODS:
        add("| " + first_day.isoformat() + " | " + last_day.isoformat() + " | "
            + reason + " |")
    add()

    add("### Seasonal effects")
    add()
    seasonal = []
    for product in state["catalogue"].values():
        if product["season"] is not None:
            seasonal.append(product["name"] + " (" + season_text(
                product["season"]) + ")")
    add("- Summer (" + months_text(SUMMER_MONTHS) + "): cooking probability x"
        + str(SUMMER_COOK_FACTOR) + ", eating out x"
        + str(SUMMER_EAT_OUT_FACTOR) + "; dishes with season \"summer\" "
        "are favoured.")
    add("- Winter (" + months_text(WINTER_MONTHS) + "): dishes with season "
        "\"winter\" (soups, casseroles) are favoured.")
    add("- Seasonal products, bought only when sold: " + ", ".join(seasonal)
        + ".")
    add("- Christmas: Christmas foods (" + ", ".join(CHRISTMAS_FOODS)
        + ") are eaten as a \"festive plate\". Between "
        + month_day_text(CHRISTMAS_PERIOD[0]) + " and "
        + month_day_text(CHRISTMAS_PERIOD[1]) + " each restocked Christmas "
        "food gets an extra package with " + pct(P_CHRISTMAS_EXTRA_PACKAGE)
        + " and the cooking probability is x" + str(CHRISTMAS_COOK_FACTOR)
        + ".")
    add()

    add("### Personal habits")
    add()
    add("Stable habits of this person, the same all year. They are not in the "
        "output files: an app could only learn their effect from the history.")
    add()
    place_parts = []
    for place in PLACE_FORGET_FACTOR:
        place_parts.append(place + " x" + str(PLACE_FORGET_FACTOR[place]))
    add("- Where products are kept (column Place in the product table; "
        "leftovers boxes: " + LEFTOVERS_PLACE + "). Food in sight gets used, "
        "food at the back of a shelf or in the crisper drawer is forgotten. "
        "Factor on the daily probability of overlooking an item: "
        + ", ".join(place_parts) + ".")
    personal_parts = []
    for name in PERSONAL_FORGET_FACTOR:
        personal_parts.append(name + " x" + str(PERSONAL_FORGET_FACTOR[name]))
    add("- Personal blind spots, factor on the daily probability of "
        "overlooking the product: " + ", ".join(personal_parts) + ". Daily "
        "staples are never overlooked; products bought for one dish are often "
        "forgotten after it.")
    appeal_parts = []
    for name in HABIT_PRODUCT_APPEAL:
        appeal_parts.append(name + " " + str(HABIT_PRODUCT_APPEAL[name]))
    add("- Habit purchases: bought out of routine or good intentions at every "
        "weekly shop with their restock probability, even when some is at "
        "home, but rarely chosen when something tastier is available. Appeal "
        "when choosing what to eat (1 = normal): " + ", ".join(appeal_parts)
        + ".")
    add()

    add("### Waste tendencies")
    add()
    add("- An item is wasted only when it is not eaten by its last good day "
        "(expiry date, or end of the opened shelf life if earlier).")
    add("- Each morning an item is overlooked for the day with probability = "
        "category base (table below) x place factor x personal factor (see "
        "Personal habits), x" + str(OPENED_FORGET_FACTOR) + " for opened "
        "items. Overlooked items are not eaten and not counted when "
        "shopping, which also leads to buying duplicates.")
    add("- Taking an item: first the product (see Snacking), then among "
        "packages of it an opened one with " + pct(P_TAKE_OPENED_FIRST)
        + ", then the one closest to its last good day with "
        + pct(P_TAKE_OLDEST_FIRST) + ", otherwise a random one. Similar items "
        "share the same meals, so with more of them in the fridge each one "
        "is more likely to spoil.")
    add("- Spoiled items are thrown away the day after their last good day, "
        "or with " + pct(P_NOTICE_LATE) + " noticed late and thrown away "
        + range_text(LATE_NOTICE_DAYS) + " days after it.")
    add()
    add("| Category | Base probability of being overlooked per day |")
    add("|---|---|")
    for category in FORGET_PROBABILITY:
        add("| " + category + " | " + pct(FORGET_PROBABILITY[category]) + " |")
    add()

    add("### Noise")
    add()
    add("- Remaining shelf life at purchase: +- " + pct(SHELF_LIFE_NOISE_SHARE)
        + " of the shelf life (rounded, at least 1 day).")
    add("- Opened shelf life: +- " + str(OPENED_LIFE_NOISE_DAYS) + " day(s), "
        "at least 1 day.")
    add("- Leftover portions: +- " + str(LEFTOVER_PORTIONS_NOISE) + ".")
    add("- All daily decisions (meals, snacks, shopping, dish choice, "
        "overlooking, noticing) are random draws with the probabilities "
        "above, so the patterns are not deterministic.")
    add()

    add("### Target waste rate")
    add()
    add("The parameters were tuned for an overall waste rate of "
        + pct(TARGET_WASTE_RATE[0]) + "-" + pct(TARGET_WASTE_RATE[1])
        + " of items; the generated data has " + rate_text(stats["wasted"],
                                                          stats["items"])
        + ". Reasoning:")
    add()
    add("- Finnish households throw away about 23 kg of avoidable food per "
        "person per year, about 4-5 % of the food they buy by weight; the "
        "largest groups are vegetables, home-cooked food (leftovers) and "
        "milk products (Silvennoinen et al. 2014, British Food Journal; "
        "Silvennoinen 2020, Luke). The main reasons are mould and passed "
        "best-before or use-by dates.")
    add("- One-person households waste more food per person than larger "
        "households.")
    add("- The per-item rate here is higher than the share by weight, because "
        "only perishable fridge items are included (no pantry or frozen "
        "food) and an item counts as wasted even when only its last part is "
        "thrown away. Roughly one fridge item in five to eight ending "
        "(at least partly) in the bin is therefore plausible.")
    add()

    add("## Limitations")
    add()
    add("- One simulated person with fixed habits; the parameters are "
        "assumptions informed by research, not fitted to real data. Real "
        "habits change over time (e.g. after noticing repeated waste).")
    add("- The outcome is per item (eaten or wasted), not the amount wasted; "
        "a portion is an abstract use, not a weight.")
    add("- Not simulated: freezing, taking food along on trips, guests, "
        "sharing food, illness, pantry and frozen food, and the fridge app "
        "itself (reminders would change behaviour).")
    add("- Spoilage follows the dates exactly. Real people often eat food "
        "after the best-before date, or throw it away before it, based on "
        "smell and look.")
    add("- Shelf lives are fixed per product with simple noise; temperature, "
        "brands and package sizes are not modelled.")
    add("- Shops, prices and offers are simplified; public holidays are only "
        "included through the listed trips.")
    add("- Real app data would contain missed scans, wrong dates and items "
        "removed without a record; this data is complete and clean.")
    add("- Items still in the fridge on the last day are left out ("
        + str(still_in_fridge) + " items), which slightly under-represents "
        "long-lasting items added near the end.")
    add()

    add("## Regenerating the data")
    add()
    add("Random seed: " + str(RANDOM_SEED) + ". Run `python " + SCRIPT_NAME
        + "` (Python 3.7+, standard library only) in the folder where the "
        "files should be written. The same seed and constants always give "
        "identical files. Change RANDOM_SEED for another sample with the same "
        "behaviour. If `" + PRODUCTS_FILE + "` is present, only the barcodes "
        "change.")

    with open(README_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


# ===========================================================================
# MAIN
# ===========================================================================

def write_json(file_name, data):
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main():
    random.seed(RANDOM_SEED)
    catalogue = build_catalogue()
    dishes = build_dishes()
    real_barcodes = assign_barcodes(catalogue, dishes)

    state = {
        "catalogue": catalogue,
        "dishes": dishes,
        "fridge": [],
        "items": [],
        "cooking_log": [],
        "last_cooked": {},
    }
    simulate(state)

    history, still_in_fridge = build_history(state)
    dish_records = build_dish_records(state)
    catalogue_records = build_catalogue_records(state)
    check_consistency(history, dish_records, catalogue_records)

    write_json(HISTORY_FILE, history)
    write_json(DISHES_FILE, dish_records)
    write_json(CATALOGUE_FILE, catalogue_records)
    write_readme(history, dish_records, state, still_in_fridge, real_barcodes)

    wasted = 0
    for record in history:
        if record["outcome"] == "wasted":
            wasted += 1
    print("Wrote " + str(len(history)) + " items (waste rate "
          + rate_text(wasted, len(history)) + "), " + str(len(dish_records))
          + " dishes, " + str(len(catalogue_records)) + " catalogue products and "
          + README_FILE + ".")


if __name__ == "__main__":
    main()
