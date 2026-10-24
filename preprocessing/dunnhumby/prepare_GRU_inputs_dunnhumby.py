import numpy as np
import sparse
import torch
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd

from IPython.display import display, clear_output
from ipywidgets import Output

from model_functions import iter_loadtxt


# ============================================================
# 1. LOAD DATA
# ============================================================

print("1. Loading Dunnhumby data...")

dunnhumby = iter_loadtxt(
    "transaction_data.csv",
    delimiter=",",
    skiprows=1
).astype("int64")

print("Original transactions:", len(dunnhumby))


# ============================================================
# 2. FILTER PRODUCTS
# Keep products appearing at least 50 times
# ============================================================

print("\n2. Filtering products...")

products = dunnhumby[:, 3]

prod_freq = np.unique(
    products,
    return_counts=True
)

prods_unique = prod_freq[0]
prods_freq = prod_freq[1]

prod_select = prods_unique[
    prods_freq >= 50
]

dunnhumby_subset = dunnhumby[
    np.isin(dunnhumby[:, 3], prod_select),
    :
]

perc_lost = len(dunnhumby_subset) / len(dunnhumby)

print("Products retained:", len(prod_select))
print("Transactions retained:", len(dunnhumby_subset))


# ============================================================
# 3. FILTER HOUSEHOLDS
# Keep households with at least 3 baskets
# ============================================================

print("\n3. Filtering households...")

all_orders = np.unique(
    dunnhumby_subset[:, 0:2],
    axis=0
)

household_freq = np.unique(
    all_orders[:, 0],
    return_counts=True
)

households_unique = household_freq[0]
households_freq = household_freq[1]

household_select = households_unique[
    households_freq >= 3
]

dunnhumby_subset = dunnhumby_subset[
    np.isin(
        dunnhumby_subset[:, 0],
        household_select
    ),
    :
]

perc_lost = len(dunnhumby_subset) / len(dunnhumby)

print("Households retained:", len(household_select))
print("Transactions retained:", len(dunnhumby_subset))


# ============================================================
# 4. REMAP HOUSEHOLD IDS
# Optimized equivalent of original GitHub loop
# ============================================================

print("\n4. Remapping household IDs...")

households, new_households = np.unique(
    dunnhumby_subset[:, 0],
    return_inverse=True
)

n_households = len(households)

dunnhumby_subset[:, 0] = new_households

print("Households:", n_households)


# ============================================================
# 5. REMAP PRODUCT IDS
# Optimized equivalent of original GitHub loop
# ============================================================

print("\n5. Remapping product IDs...")

products, new_products = np.unique(
    dunnhumby_subset[:, 3],
    return_inverse=True
)

n_products = len(products)

dunnhumby_subset[:, 3] = new_products

print("Products:", n_products)


# ============================================================
# 6. SORT TRANSACTIONS
# household -> day -> transaction time
# ============================================================

print("\n6. Sorting transactions...")

order = np.lexsort((
    dunnhumby_subset[:, 8],
    dunnhumby_subset[:, 2],
    dunnhumby_subset[:, 0]
))

dunnhumby_ord = dunnhumby_subset[order, :]


# ============================================================
# 7. CREATE SEQUENTIAL BASKET NUMBERS
# ============================================================

print("\n7. Creating basket sequences...")

n_purch = len(dunnhumby_ord)

new_basket = np.full(n_purch, False)
new_household = np.full(n_purch, False)

new_basket[1:] = (
    dunnhumby_ord[1:n_purch, 1]
    !=
    dunnhumby_ord[0:(n_purch - 1), 1]
)

new_household[1:] = (
    dunnhumby_ord[1:n_purch, 0]
    !=
    dunnhumby_ord[0:(n_purch - 1), 0]
)

times = np.zeros(n_purch).astype("int64")

t = 0

for i in np.arange(n_purch):

    if new_household[i]:
        t = 0

    elif new_basket[i]:
        t += 1

    times[i] = t

dunnhumby_ord[:, 1] = times


# ============================================================
# 8. LIMIT TO FIRST 100 BASKETS
# ============================================================

print("\n8. Limiting basket history...")

maxlen = 100

dunnhumby_trunc = dunnhumby_ord[
    dunnhumby_ord[:, 1] < maxlen,
    :
]

n_purch = len(dunnhumby_trunc)

n_bask_perh = np.zeros(
    n_households
).astype("int64")

for h in np.arange(n_households):

    n_bask_perh[h] = (
        np.max(
            dunnhumby_ord[
                dunnhumby_ord[:, 0] == h,
                1
            ]
        ) + 1
    )

perc_baskets_removed = np.mean(
    n_bask_perh > maxlen
)


# ============================================================
# 9. CREATE COORDINATES
# household, basket, product
# ============================================================

print("\n9. Creating purchase coordinates...")

dunnhumby_coord = dunnhumby_trunc[
    :,
    np.array([0, 1, 3])
]


# ============================================================
# 10. CREATE ORDER INFORMATION
# ============================================================

first_of_basket = np.full(
    n_purch,
    True
)

first_of_basket[1:] = (
    dunnhumby_trunc[1:n_purch, 1]
    !=
    dunnhumby_trunc[0:(n_purch - 1), 1]
)

orders = dunnhumby_trunc[
    first_of_basket,
    :
]

dunnhumby_orders = orders[
    :,
    np.array([0, 1, 2, 6, 8, 9])
]


# ============================================================
# 11. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n10. Creating train/validation/test split...")

nusers = np.max(
    dunnhumby_coord[:, 0]
) + 1

np.random.seed(123)

test_users = np.sort(
    np.random.choice(
        np.arange(nusers),
        int(np.floor(nusers / 2)),
        replace=False
    )
)

orders_set = np.zeros(
    len(dunnhumby_orders)
).astype("int64")

last_of_user = np.full(
    len(dunnhumby_orders),
    True
)

last_of_user[
    0:(len(last_of_user) - 1)
] = (
    dunnhumby_orders[
        0:(len(last_of_user) - 1),
        0
    ]
    !=
    dunnhumby_orders[
        1:len(last_of_user),
        0
    ]
)

is_user_in_test = np.isin(
    dunnhumby_orders[:, 0],
    test_users
)

orders_set[
    last_of_user & ~is_user_in_test
] = 1

orders_set[
    last_of_user & is_user_in_test
] = 2

orders_set = np.expand_dims(
    orders_set,
    1
)

order_ids = np.expand_dims(
    np.arange(len(orders_set)),
    1
)

dunnhumby_orders = np.concatenate(
    (
        order_ids,
        dunnhumby_orders,
        orders_set
    ),
    1
)


# ============================================================
# 12. ASSIGN ORDER IDS
# ============================================================

print("\n11. Assigning order IDs...")

order_id = np.zeros(
    len(dunnhumby_coord)
).astype("int64")

count = 0

new_order = np.full(
    len(dunnhumby_coord),
    False
)

new_order[1:] = (
    dunnhumby_coord[1:, 1]
    !=
    dunnhumby_coord[
        0:(len(dunnhumby_coord) - 1),
        1
    ]
)

for i in np.arange(
    len(dunnhumby_coord)
):

    if new_order[i]:
        count += 1

    order_id[i] = count

purchase_set = np.expand_dims(
    dunnhumby_orders[
        order_id,
        7
    ],
    1
)

order_id = np.expand_dims(
    order_id,
    1
)

dunnhumby_coord = np.concatenate(
    (
        dunnhumby_coord,
        order_id,
        purchase_set
    ),
    1
)


# ============================================================
# 13. OPTIONAL TARS-LIKE TRUNCATION
# ============================================================

trunc = False

if trunc:

    maxlen = 51

    nusers = np.max(
        dunnhumby_coord[:, 0]
    ) + 1

    ntime = np.max(
        dunnhumby_coord[:, 1]
    ) + 1

    nprod = np.max(
        dunnhumby_coord[:, 2]
    ) + 1

    train_coordinates = dunnhumby_coord[
        dunnhumby_coord[:, 4] == 0,
        0:3
    ]

    train_sequences = sparse.COO(
        np.ndarray.transpose(
            train_coordinates
        ),
        True,
        shape=(
            nusers,
            ntime - 1,
            nprod
        )
    )

    complete_last = np.sum(
        np.sum(
            train_sequences,
            axis=2
        ) > 0,
        axis=1
    ).todense()

    trunc_coord = np.zeros(
        (0, 5)
    ).astype("int64")

    for i in np.arange(nusers):

        selection = (
            dunnhumby_coord[:, 0] == i
        )

        old_coords = dunnhumby_coord[
            selection,
            :
        ]

        n_baskets = (
            complete_last[i] + 1
        )

        if n_baskets > maxlen:

            select_mostrecent = (
                old_coords[:, 1]
                >=
                n_baskets - maxlen
            )

            old_coords = old_coords[
                select_mostrecent
            ]

            old_coords[:, 1] = (
                old_coords[:, 1]
                -
                (n_baskets - maxlen)
            )

        trunc_coord = np.concatenate(
            (
                trunc_coord,
                old_coords
            ),
            0
        )

        if i % 100 == 0:
            print(i)

    dunnhumby_coord = trunc_coord


# ============================================================
# 14. TRAIN / VALIDATION / TEST COORDINATES
# ============================================================

print("\n12. Creating sparse GRU inputs...")

train_orders = dunnhumby_orders[
    dunnhumby_orders[:, 7] == 0,
    :
]

test_orders = dunnhumby_orders[
    dunnhumby_orders[:, 7] == 1,
    :
]

val_orders = dunnhumby_orders[
    dunnhumby_orders[:, 7] == 2,
    :
]

train_coordinates = dunnhumby_coord[
    dunnhumby_coord[:, 4] == 0,
    0:3
]

test_coordinates = dunnhumby_coord[
    dunnhumby_coord[:, 4] == 1,
    0:3
]

val_coordinates = dunnhumby_coord[
    dunnhumby_coord[:, 4] == 2,
    0:3
]

nusers = np.max(
    dunnhumby_coord[:, 0]
) + 1

ntime = np.max(
    dunnhumby_coord[:, 1]
) + 1

nprod = np.max(
    dunnhumby_coord[:, 2]
) + 1


# ============================================================
# 15. TRAINING SEQUENCES
# ============================================================

train_sequences = sparse.COO(
    np.ndarray.transpose(
        train_coordinates
    ),
    True,
    shape=(
        nusers,
        ntime - 1,
        nprod
    )
)

train_last = np.sum(
    np.sum(
        train_sequences,
        axis=2
    ) > 0,
    axis=1
).todense() - 1

is_last = (
    train_coordinates[:, 1]
    ==
    train_last[
        train_coordinates[:, 0]
    ]
)

train_input = sparse.COO(
    np.ndarray.transpose(
        train_coordinates[
            is_last == False
        ]
    ),
    True,
    shape=(
        nusers,
        ntime - 2,
        nprod
    )
)

is_first = (
    train_coordinates[:, 1] == 0
)

train_output = sparse.COO(
    np.ndarray.transpose(
        train_coordinates[
            is_first == False
        ]
    ),
    True,
    shape=(
        nusers,
        ntime - 1,
        nprod
    )
)

train_output = train_output[
    :,
    1:(ntime - 1),
    :
]


# ============================================================
# 16. COMPLETE INPUT / OUTPUT
# ============================================================

complete_last = train_last + 1

is_last = (
    dunnhumby_coord[:, 1]
    ==
    complete_last[
        dunnhumby_coord[:, 0]
    ]
)

complete_input = sparse.COO(
    np.ndarray.transpose(
        dunnhumby_coord[
            is_last == False,
            0:3
        ]
    ),
    True,
    shape=(
        nusers,
        ntime - 1,
        nprod
    )
)

is_first = (
    dunnhumby_coord[:, 1] == 0
)

complete_output = sparse.COO(
    np.ndarray.transpose(
        dunnhumby_coord[
            is_first == False,
            0:3
        ]
    ),
    True,
    shape=(
        nusers,
        ntime,
        nprod
    )
)

complete_output = complete_output[
    :,
    1:ntime,
    :
]


# ============================================================
# 17. VALIDATION / TEST INPUT
# ============================================================

val_users = np.unique(
    val_coordinates[:, 0]
)

test_users = np.unique(
    test_coordinates[:, 0]
)

val_input = complete_input[
    val_users,
    :,
    :
]

test_input = complete_input[
    test_users,
    :,
    :
]

val_output = complete_output[
    val_users,
    :,
    :
]

test_output = complete_output[
    test_users,
    :,
    :
]

val_last = complete_last[
    val_users
]

test_last = complete_last[
    test_users
]


# ============================================================
# 18. TIME FEATURES
# ============================================================

print("\n13. Creating time features...")

hour_of_day = np.floor(
    dunnhumby_orders[:, 5] / 100
).astype("int64")

night = np.array(
    [23, 0, 1, 2, 3, 4, 5]
)

morning = np.array(
    [6, 7]
)

daytime = np.arange(
    8,
    21
)

evening = np.array(
    [21, 22]
)

time_of_day = hour_of_day.copy()

time_of_day[
    np.isin(hour_of_day, night)
] = 0

time_of_day[
    np.isin(hour_of_day, morning)
] = 1

time_of_day[
    np.isin(hour_of_day, daytime)
] = (
    hour_of_day[
        np.isin(hour_of_day, daytime)
    ]
    - 6
)

time_of_day[
    np.isin(hour_of_day, evening)
] = 15

purchase_day = dunnhumby_orders[:, 3]

day_of_week = purchase_day % 7

time_of_week = (
    day_of_week * 16
    +
    time_of_day
)

time_of_week = np.expand_dims(
    time_of_week,
    1
)


# ============================================================
# 19. DAYS SINCE PRIOR
# ============================================================

first_of_user = np.full(
    len(dunnhumby_orders),
    True
)

first_of_user[1:] = (
    dunnhumby_orders[1:, 1]
    !=
    dunnhumby_orders[
        0:(len(first_of_user) - 1),
        1
    ]
)

days_since_prior = np.zeros(
    len(dunnhumby_orders)
).astype("int64")

days_since_prior[1:] = (
    dunnhumby_orders[1:, 3]
    -
    dunnhumby_orders[
        0:(len(days_since_prior) - 1),
        3
    ]
)

days_since_prior[
    first_of_user
] = 0

capped_days_since_prior = np.copy(
    days_since_prior
)

capped_days_since_prior[
    days_since_prior > 50
] = 50


# ============================================================
# 20. CUMULATIVE DAYS
# ============================================================

cumdays_since_first = np.zeros(
    len(days_since_prior)
).astype("int64")

total = 0

for i in np.arange(
    1,
    len(days_since_prior)
):

    if (
        dunnhumby_orders[i, 1]
        !=
        dunnhumby_orders[i - 1, 1]
    ):

        total = 0

    else:

        total += days_since_prior[i]

        cumdays_since_first[i] = total


# ============================================================
# 21. MONTH FEATURES
# ============================================================

day_of_year = (
    dunnhumby_orders[:, 3] % 365
)

month_of_year = np.floor(
    day_of_year / (365 / 12)
).astype("int64")

rel_timestamp = (
    dunnhumby_orders[:, 2:3]
    /
    np.max(
        dunnhumby_orders[:, 2:3]
    )
)

capped_days_since_prior = (
    np.expand_dims(
        capped_days_since_prior,
        1
    )
    /
    np.max(capped_days_since_prior)
)

cumdays_since_first = (
    np.expand_dims(
        cumdays_since_first,
        1
    )
    /
    np.max(cumdays_since_first)
)

month_of_year_pd = pd.DataFrame(
    month_of_year
)

month_dummies = pd.get_dummies(
    month_of_year_pd.astype(str)
).to_numpy()

month_dummies = month_dummies[
    :,
    [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 2]
]


# ============================================================
# 22. SEQUENCE VARIABLES
# ============================================================

seq_variables = np.concatenate(
    (
        rel_timestamp,
        capped_days_since_prior,
        cumdays_since_first,
        month_dummies
    ),
    1
)

seq_dim = seq_variables.shape[1]

seq_input = np.zeros(
    (
        nusers,
        ntime - 1,
        seq_dim
    )
)

for user in np.arange(nusers):

    seq = seq_variables[
        dunnhumby_orders[:, 0] == user,
        :
    ]

    seq = seq[
        seq[:, 0] > 0,
        :
    ]

    seq_input[
        user,
        np.arange(seq.shape[0]),
        :
    ] = seq


# ============================================================
# 23. HOUR INPUT
# ============================================================

hour_dim = 111

hour_input = np.concatenate(
    (
        dunnhumby_orders[:, 1:3],
        time_of_week
    ),
    1
)

train_hour = hour_input[
    dunnhumby_orders[:, 7] == 0,
    :
]

train_hour = train_hour[
    train_hour[:, 2] < hour_dim,
    :
]

train_hour_input = sparse.COO(
    np.ndarray.transpose(
        train_hour
    ),
    True,
    shape=(
        nusers,
        ntime - 1,
        hour_dim
    )
)

train_hour_input = train_hour_input[
    :,
    1:(ntime - 1),
    :
]


# ============================================================
# 24. TRAIN SEQUENCE INPUT
# ============================================================

train_seq_input = np.copy(
    seq_input
)

train_seq_input[
    np.arange(nusers),
    train_last,
    :
] = 0

train_seq_input = train_seq_input[
    :,
    0:(ntime - 2),
    :
]

val_seq_input = seq_input[
    val_users,
    :,
    :
]

test_seq_input = seq_input[
    test_users,
    :,
    :
]


# ============================================================
# 25. VALIDATION HOUR INPUT
# ============================================================

val_hour = hour_input[
    dunnhumby_orders[:, 7] == 2,
    :
]

val_hour = val_hour[
    val_hour[:, 2] < hour_dim,
    :
]

val_hour_input = sparse.COO(
    np.ndarray.transpose(
        val_hour
    ),
    True,
    shape=(
        nusers,
        ntime,
        hour_dim
    )
)

val_hour_input = val_hour_input[
    val_users,
    1:ntime,
    :
]


# ============================================================
# 26. TEST HOUR INPUT
# ============================================================

test_hour = hour_input[
    dunnhumby_orders[:, 7] == 1,
    :
]

test_hour = test_hour[
    test_hour[:, 2] < hour_dim,
    :
]

test_hour_input = sparse.COO(
    np.ndarray.transpose(
        test_hour
    ),
    True,
    shape=(
        nusers,
        ntime,
        hour_dim
    )
)

test_hour_input = test_hour_input[
    test_users,
    1:ntime,
    :
]


print("\n==========================================")
print("Dunnhumby preprocessing completed!")
print("Users:", nusers)
print("Products:", nprod)
print("Maximum sequence length:", ntime)
print("==========================================")
# ============================================================
# SAVE PREPROCESSED DATA
# ============================================================

import pickle
import os

print("\nSaving preprocessed Dunnhumby data...")

os.makedirs("processed", exist_ok=True)

processed_data = {
    # Main training data
    "train_input": train_input,
    "train_output": train_output,

    # Validation data
    "val_input": val_input,
    "val_output": val_output,

    # Test data
    "test_input": test_input,
    "test_output": test_output,

    # Sequence/time covariates
    "train_seq_input": train_seq_input,
    "val_seq_input": val_seq_input,
    "test_seq_input": test_seq_input,

    "train_hour_input": train_hour_input,
    "val_hour_input": val_hour_input,
    "test_hour_input": test_hour_input,

    # Last basket information
    "train_last": train_last,
    "val_last": val_last,
    "test_last": test_last,

    # Orders
    "train_orders": train_orders,
    "val_orders": val_orders,
    "test_orders": test_orders,

    # Dataset dimensions
    "nusers": nusers,
    "ntime": ntime,
    "nprod": nprod,
    "seq_dim": seq_dim,
    "hour_dim": hour_dim
}

with open(
    "processed/dunnhumby_preprocessed.pkl",
    "wb"
) as f:
    pickle.dump(
        processed_data,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("Saved successfully!")
print("File: processed/dunnhumby_preprocessed.pkl")
