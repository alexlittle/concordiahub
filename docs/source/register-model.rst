Registering a Model with ``concordiahub``
=========================================

This guide walks through how to register a model with ``concordiahub`` so it can be
evaluated and compared alongside other models in the framework.

Divide into 2 sections one for each of:

#. **Standard models** - scikit-learn, XGBoost, or similar, where ``concordiahub``
   runs the XAI method itself.
#. **LLM-based models** - where explanations are pre-generated and cached, and
   ``concordiahub`` only scores them.

Then with the steps to build it up with code snippets and a final full example
for each

