QuickStart
--------------

Install the package::

    pip install xailens

Add at the end of your model model code:

.. code-block:: python

    import xailens

    # the data/predictions/model you need for running XAI Lens
    # probably these are already in your code, this is just for info as to what you need
    raw = pd.read_csv('raw_data.csv') # your raw dataset (for test instances at least)
    X_test = pd.read_csv('X_test.csv') # your test instances
    y_pred = pd.read_csv('y_pred.csv') # your models predictions
    y_test = pd.read_csv('y_test.csv') # your test targets
    model = joblib.load('model.joblib') # your model

    # define the dataschema
    data_schema = xailens.DataSchema(
        instance_id_column="instance_uuid", # the unique identifier for your instances
        target_column="target", # name of your target column, from y_test
        prediction_column="y_pred" # name of your prediction column, from y_pred
        )

    # your data and predictions
    model_data = xailens.ModelData(
        data_schema=data_schema, # DataSchema from above
        raw_instances=raw, # your raw data
        x_test=X_test, # your X_test dataframe
        y_pred=y_pred, # your y_pred dataframe
        y_test=y_test # your y_test dataframe
        )

    # overall context
    ctx = xailens.ModelContext(
        "logistic_regression", # model type, can be "logistic_regression", "xgboost" or "llm"
        ["coefficients"], # XAI methods to apply, can be "coefficients", "tree_shap" or "llm_reasoning"
        "cleveland-heart-data", # identifier for your dataset
        model_data, # ModelData from above
        model=model, # your model (joblib file)
        model_dir_name="heart_lr", # directory name to save to (can be blank)
        model_display_name="Logistic Regression" # display name for your model, as it will appear in the dashboard
        )

    xailens.run(ctx)


Start the dashboard::

    xailens-dash

