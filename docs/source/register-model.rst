Registering a Model with ``xailens``
===================================

This guide walks through how to register a model with ``xailens`` so it can be
evaluated and compared alongside other models in the framework.

Divide into 2 sections one for each of:

#. **Standard models** - scikit-learn, XGBoost, or similar, where ``xailens``
   runs the XAI method itself.
#. **LLM-based models** - where explanations are pre-generated and cached, and
   ``xailens`` only scores them.

Then with the steps to build it up with code snippets and a final full example
for each

