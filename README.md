# Market Basket Analysis & Next-Basket Prediction

A group project focused on analyzing customer purchasing behavior and predicting products that customers are likely to purchase in their next basket.

## Datasets

### Dunnhumby – The Complete Journey
Retail transaction data containing household-level purchase histories.

Dataset:
https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey

### Instacart Market Basket Analysis
Customer order histories containing products, aisles, departments and repeated orders.

Dataset:
https://www.kaggle.com/c/instacart-market-basket-analysis

## Project Structure

```text
group-project/
├── preprocessing/
│   ├── dunnhumby/
│   │   └── prepare_GRU_inputs_dunnhumby.py
│   └── instacart/
│       ├── cluster_products_instacart.R
│       └── prepare_GRU_inputs_instacart.py
├── model/
│   ├── model_functions.py
│   ├── train_dunnhumby.py
│   ├── train_dunnhumby_withcov.py
│   ├── train_instacart.py
│   └── train_instacart_withcov.py
└── eval/
