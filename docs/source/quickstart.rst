Quickstart
----------

1. Install the package

.. code-block:: console

   (venv) $ pip install xailens

2 .Add at the end of your model code:

.. code-block:: python

    import xailens

    # define the dataschema
    data_schema = xailens.DataSchema(
        # name of the unique identifier column for your instances
        instance_id_column="instance_uuid",
        # name of your target column, from y_test
        target_column="target",
        # name of your prediction column, from y_pred
        prediction_column="pred_value")

    # your data and predictions
    model_data = xailens.ModelData(
        data_schema=data_schema,        # DataSchema from above
        raw_instances=raw,              # your raw data
        x_test=X_test,                  # your X_test dataframe
        y_pred=y_pred,                  # your y_pred dataframe
        y_test=y_test)                  # your y_test dataframe

    # overall context
    ctx = xailens.ModelContext(
        # model type, can be "logistic_regression"," xgboost" or "llm"
        "logistic_regression",
        # XAI methods to apply, options are "coefficients", "tree_shap",
        # "llm_reasoning"
        ["coefficients"],
        # identifier for your dataset
        "cleveland-heart-data",
        # ModelData from above
        model_data,
        # your model object (that will be saved as joblib file)
        model=model,
        # directory name to save to (can be blank)
        model_dir_name="heart_lr",
        # display name for your model, as it will appear in the dashboard
        model_display_name="Logistic Regression")

    xailens.run(ctx)


3. Start the dashboard

.. code-block:: console

   (venv) $ xailens-dash

